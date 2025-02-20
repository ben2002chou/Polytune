import os
import mir_eval
import glob
import pretty_midi
import numpy as np
import librosa
import note_seq
import collections
import concurrent.futures
import traceback
import tempfile
from tqdm import tqdm


# def get_granular_program(program_number, is_drum, granularity_type):
#     """
#     Returns the granular program number based on the given parameters.

#     Parameters:
#     program_number (int): The original program number.
#     is_drum (bool): Indicates whether the program is a drum program or not.
#     granularity_type (str): The type of granularity to apply.

#     Returns:
#     int: The granular program number.

#     """
#     if granularity_type == "full":
#         return program_number
#     elif granularity_type == "midi_class":
#         return (program_number // 8) * 8
#     elif granularity_type == "flat":
#         return 0 if not is_drum else 1


# Standard evaluation Pipeline for MT3
def compute_transcription_metrics(ref_mid, est_mid):
    """Helper function to compute onset/offset, onset only, and frame metrics."""
    ns_ref = note_seq.midi_file_to_note_sequence(ref_mid)
    ns_est = note_seq.midi_file_to_note_sequence(est_mid)
    intervals_ref, pitches_ref, _ = note_seq.sequences_lib.sequence_to_valued_intervals(
        ns_ref
    )
    intervals_est, pitches_est, _ = note_seq.sequences_lib.sequence_to_valued_intervals(
        ns_est
    )
    len_est_intervals = len(intervals_est)
    len_ref_intervals = len(intervals_ref)

    # onset-offset
    onoff_precision, onoff_recall, onoff_f1, onoff_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            intervals_ref, pitches_ref, intervals_est, pitches_est
        )
    )

    # onset-only
    on_precision, on_recall, on_f1, on_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            intervals_ref, pitches_ref, intervals_est, pitches_est, offset_ratio=None
        )
    )

    return {
        "len_ref_intervals": len_ref_intervals,
        "len_est_intervals": len_est_intervals,
        "onoff_precision": onoff_precision,
        "onoff_recall": onoff_recall,
        "onoff_f1": onoff_f1,
        "onoff_overlap": onoff_overlap,
        "on_precision": on_precision,
        "on_recall": on_recall,
        "on_f1": on_f1,
        "on_overlap": on_overlap,
    }


# This is multi-instrument F1 score
def mt3_program_aware_note_scores(fextra, fremoved, fcorrect, fextra_gt, fremoved_gt, fcorrect_gt, granularity):
    """
    Edited version of MT3's program aware precision/recall/F1 score.
    We follow Perceiver's evaluation approach which takes only onset and program into account.
    Using MIDIs transcribed from MT3, we managed to get similar results as Perceiver, which is 0.75 for onset F1.
    """

    # List of MIDI files to combine
    midi_files = [fextra, fremoved, fcorrect]
    midi_files_gt = [fextra_gt, fremoved_gt, fcorrect_gt]

    # Create a new PrettyMIDI object for the combined MIDI
    combined_midi = pretty_midi.PrettyMIDI()
    combined_midi_gt = pretty_midi.PrettyMIDI()

    # Load each file and add its instruments to the combined MIDI
    for midi_file in midi_files:
        # Load the current MIDI file
        current_midi = pretty_midi.PrettyMIDI(midi_file)
        
        # Add each instrument from the current file to the combined MIDI file
        for instrument in current_midi.instruments:
            combined_midi.instruments.append(instrument)   
    est_mid = combined_midi # reference midi (music score) 
    
    for midi_file in midi_files_gt:
        # Load the current MIDI file
        current_midi = pretty_midi.PrettyMIDI(midi_file)
        
        # Add each instrument from the current file to the combined MIDI file
        for instrument in current_midi.instruments:
            combined_midi_gt.instruments.append(instrument)
    est_mid_gt = combined_midi_gt # reference midi (music score)
    # Print details about each instrument (treated as a track)
    for index, instrument in enumerate(est_mid.instruments):
        is_drum = 'Drum' if instrument.is_drum else 'Not Drum'
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print(f"Track {index}: Instrument {instrument.program}, Name: {instrument_name}, {is_drum}")
        
    for index, instrument in enumerate(est_mid_gt.instruments):
        is_drum = 'Drum' if instrument.is_drum else 'Not Drum'
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print(f"Track {index}: Instrument {instrument.program}, Name: {instrument_name}, {is_drum}")
        
    # Write the combined MIDI to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_midi_file:
        combined_midi.write(temp_midi_file.name)
        temp_midi_file_path = temp_midi_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_midi_file_gt:
        combined_midi_gt.write(temp_midi_file_gt.name)
        temp_midi_file_path_gt = temp_midi_file_gt.name
    res = dict() # results
    est_ns = note_seq.midi_file_to_note_sequence(temp_midi_file_path)
    
    ref_ns = note_seq.midi_file_to_note_sequence(temp_midi_file_path_gt)
    # TODO: We might need to remove drums and process separately as in MT3
    # NOTE: We don't need to remove drums and process separately as in MT3
    # as we consider onset only for all instruments.
    # def remove_drums(ns):
    #   ns_drumless = note_seq.NoteSequence()
    #   ns_drumless.CopyFrom(ns)
    #   del ns_drumless.notes[:]
    #   ns_drumless.notes.extend([note for note in ns.notes if not note.is_drum])
    #   return ns_drumless

    # est_ns_drumless = remove_drums(est_ns)
    # ref_ns_drumless = remove_drums(ref_ns)

    est_tracks = [est_ns]
    ref_tracks = [ref_ns]
    use_track_offsets = [False]
    use_track_velocities = [False]
    track_instrument_names = [""]
    
    # important!!!
    # this part calculates instrument-agnostic onset F1 score
    # it is the same as: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L255
    for est_ns, ref_ns, use_offsets, use_velocities, instrument_name in zip(
        est_tracks,
        ref_tracks,
        use_track_offsets,
        use_track_velocities,
        track_instrument_names,
    ):

        est_intervals, est_pitches, est_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
        )
        # intervals is like (note.start_time, note.end_time)

        ref_intervals, ref_pitches, ref_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)
        )

        # Precision / recall / F1 using onsets (and pitches) only.
        # looks like we can just do this seperately for each type of error!
        precision, recall, f_measure, avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                offset_ratio=None,
            )
        )
        res["Onset precision"] = precision
        res["Onset recall"] = recall
        res["Onset F1"] = f_measure
        
        
        
    # Iterate over each estimated track and corresponding reference track
    for track_index, (ref_mid, est_instrument) in enumerate(zip(midi_files_gt, est_mid.instruments)):
        # Skip if estimated track has more instruments than the reference tracks
        

        
        ref_ns = note_seq.midi_file_to_note_sequence(ref_mid)

        # Write the estimated instrument to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mid') as est_temp_midi_file:
            est_temp_mid = pretty_midi.PrettyMIDI()
            est_temp_mid.instruments.append(est_instrument)
            est_temp_mid.write(est_temp_midi_file.name)
            est_ns = note_seq.midi_file_to_note_sequence(est_temp_midi_file.name)

        # Calculate evaluation metrics
        est_intervals, est_pitches, _ = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
        )
        ref_intervals, ref_pitches, _ = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)
        )

        precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            offset_ratio=None
        )

        # Store results for this track
        res[f"Track {track_index} precision"] = precision
        res[f"Track {track_index} recall"] = recall
        res[f"Track {track_index} F1"] = f_measure

    # # group notes by program number
    # ref_inst_to_notes_mapping = {}
    # est_inst_to_notes_mapping = {}

    # # this part calculates multi-instrument onset F1 score
    # # based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L36

    # # following MT3, this will group notes under the same instrument program, determined by the granularity
    # for inst in ref_mid.instruments:
    #     cur_ref_program = get_granular_program(
    #         inst.program, inst.is_drum, granularity_type
    #     )
    #     if (cur_ref_program, inst.is_drum) in ref_inst_to_notes_mapping:
    #         ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] += [
    #             note for note in inst.notes
    #         ]
    #     else:
    #         ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] = [
    #             note for note in inst.notes
    #         ]

    # for inst in est_mid.instruments:
    #     cur_est_program = get_granular_program(
    #         inst.program, inst.is_drum, granularity_type
    #     )
    #     if (cur_est_program, inst.is_drum) in est_inst_to_notes_mapping:
    #         est_inst_to_notes_mapping[(cur_est_program, inst.is_drum)] += [
    #             note for note in inst.notes
    #         ]
    #     else:
    #         est_inst_to_notes_mapping[(cur_est_program, inst.is_drum)] = [
    #             note for note in inst.notes
    #         ]

    # # this part is based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L82
    # program_and_is_drum_tuples = set(ref_inst_to_notes_mapping.keys()) | set(
    #     est_inst_to_notes_mapping.keys()
    # )
    # drum_precision_sum = 0.0
    # drum_precision_count = 0
    # drum_recall_sum = 0.0
    # drum_recall_count = 0

    # nondrum_precision_sum = 0.0
    # nondrum_precision_count = 0
    # nondrum_recall_sum = 0.0
    # nondrum_recall_count = 0

    # program_f1 = {}
    # for program, is_drum in program_and_is_drum_tuples:
    #     if (program, is_drum) in ref_inst_to_notes_mapping:
    #         ref_notes = ref_inst_to_notes_mapping[(program, is_drum)]
    #         ref_intervals = np.array([[note.start, note.end] for note in ref_notes])
    #         ref_pitches = np.array(
    #             [librosa.midi_to_hz(note.pitch) for note in ref_notes]
    #         )
    #     else:
    #         # ref does not have this instrument
    #         ref_intervals = np.zeros((0, 2))
    #         ref_pitches = np.zeros(0)

    #     if (program, is_drum) in est_inst_to_notes_mapping:
    #         est_notes = est_inst_to_notes_mapping[(program, is_drum)]
    #         est_intervals = np.array([[note.start, note.end] for note in est_notes])
    #         est_pitches = np.array(
    #             [librosa.midi_to_hz(note.pitch) for note in est_notes]
    #         )
    #     else:
    #         # est does not have this instrument
    #         est_intervals = np.zeros((0, 2))
    #         est_pitches = np.zeros(0)

    #     # NOTE: like Perceiver, disable offset calculation
    #     args = {
    #         "ref_intervals": ref_intervals,
    #         "ref_pitches": ref_pitches,
    #         "est_intervals": est_intervals,
    #         "est_pitches": est_pitches,
    #         "offset_ratio": None,
    #     }
    #     precision, recall, f_measure, unused_avg_overlap_ratio = (
    #         mir_eval.transcription.precision_recall_f1_overlap(**args)
    #     )

    #     # print(f"program={program} is_drum={is_drum} est={est_pitches.shape[0]} ref={ref_pitches.shape[0]}")
    #     # print(f"precision={precision} recall={recall} f_measure={f_measure}")
    #     # print(f"est_intervals={len(est_intervals)} ref_intervals={len(ref_intervals)}")
    #     # print("======")

    #     if granularity_type == "midi_class":
    #         if is_drum:
    #             program_f1[-1] = f_measure
    #         else:
    #             program_f1[program] = f_measure

    #     if is_drum:
    #         drum_precision_sum += precision * len(est_intervals)
    #         drum_precision_count += len(est_intervals)
    #         drum_recall_sum += recall * len(ref_intervals)
    #         drum_recall_count += len(ref_intervals)
    #     else:
    #         nondrum_precision_sum += precision * len(est_intervals)
    #         nondrum_precision_count += len(est_intervals)
    #         nondrum_recall_sum += recall * len(ref_intervals)
    #         nondrum_recall_count += len(ref_intervals)

    # precision_sum = drum_precision_sum + nondrum_precision_sum
    # precision_count = drum_precision_count + nondrum_precision_count
    # recall_sum = drum_recall_sum + nondrum_recall_sum
    # recall_count = drum_recall_count + nondrum_recall_count

    # # print(f"precision_sum={precision_sum} precision_count={precision_count}")
    # # print(f"recall_sum={recall_sum} recall_count={recall_count}")

    # precision = (precision_sum / precision_count) if precision_count else 0
    # recall = (recall_sum / recall_count) if recall_count else 0
    # f_measure = mir_eval.util.f_measure(precision, recall)

    # drum_precision = (
    #     (drum_precision_sum / drum_precision_count) if drum_precision_count else 0
    # )
    # drum_recall = (drum_recall_sum / drum_recall_count) if drum_recall_count else 0
    # drum_f_measure = mir_eval.util.f_measure(drum_precision, drum_recall)

    # nondrum_precision = (
    #     (nondrum_precision_sum / nondrum_precision_count)
    #     if nondrum_precision_count
    #     else 0
    # )
    # nondrum_recall = (
    #     (nondrum_recall_sum / nondrum_recall_count) if nondrum_recall_count else 0
    # )
    # nondrum_f_measure = mir_eval.util.f_measure(nondrum_precision, nondrum_recall)

    # res.update(
    #     {
    #         f"Onset + program precision ({granularity_type})": precision,
    #         f"Onset + program recall ({granularity_type})": recall,
    #         f"Onset + program F1 ({granularity_type})": f_measure,
    #         # f'Drum onset precision ({granularity_type})': drum_precision,
    #         # f'Drum onset recall ({granularity_type})': drum_recall,
    #         # f'Drum onset F1 ({granularity_type})': drum_f_measure,
    #         # f'Nondrum onset + program precision ({granularity_type})':
    #         #     nondrum_precision,
    #         # f'Nondrum onset + program recall ({granularity_type})':
    #         #     nondrum_recall,
    #         # f'Nondrum onset + program F1 ({granularity_type})':
    #         #     nondrum_f_measure
    #         "F1 by program": program_f1,
    #     }
    # )
    return res

# def loop_transcription_eval(ref_mid, est_mid):
#     """
#     This evaluation takes in account the separability of the model. Goes by "track" instead of tight
#     coupling to "program number". This is because of a few reasons:
#     - for loops, the program number in ref can be arbitrary
#         - e.g. how do you assign program number to Vox?
#         - no one use program number for synth / sampler etc.
#         - string contrabass VS bass midi class are different, but can be acceptable
#         - leads and key / synth pads and electric piano
#     - the "track splitting" aspect is more important than the accuracy of the midi program number
#         - we can have wrong program number, but as long as they are grouped in the correct track
#     - hence we propose 2 more evaluation metrics:
#         - f1_score_matrix for each ref_track VS est_track, take the mean of the maximum f1 score for each ref_track
#         - number of tracks
#     """
#     score_matrix = np.zeros((len(ref_mid.instruments), len(est_mid.instruments)))

#     for i, ref_inst in enumerate(ref_mid.instruments):
#         for j, est_inst in enumerate(est_mid.instruments):
#             if ref_inst.is_drum == est_inst.is_drum:
#                 ref_intervals = np.array(
#                     [[note.start, note.end] for note in ref_inst.notes]
#                 )
#                 ref_pitches = np.array(
#                     [librosa.midi_to_hz(note.pitch) for note in ref_inst.notes]
#                 )
#                 est_intervals = np.array(
#                     [[note.start, note.end] for note in est_inst.notes]
#                 )
#                 est_pitches = np.array(
#                     [librosa.midi_to_hz(note.pitch) for note in est_inst.notes]
#                 )

#                 _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
#                     ref_intervals, ref_pitches, est_intervals, est_pitches
#                 )
#                 score_matrix[i][j] = f1

#     inst_idx = np.argmax(score_matrix, axis=-1)
#     ref_progs = [inst.program for inst in ref_mid.instruments]
#     est_progs = [est_mid.instruments[idx].program for idx in inst_idx]
#     return (
#         np.mean(np.max(score_matrix, axis=-1)),
#         len(ref_mid.instruments),
#         len(est_mid.instruments),
#     )




def midi_to_note_tuples(midi_file):
    """Extract note data from a MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = [(note.pitch, note.start, note.end) for instrument in midi_data.instruments for note in instrument.notes]
    return notes

def compare_midi(gt_notes, student_notes, onset_tolerance=0.2):
    """Identify missed, correct, and extra notes based on onset and pitch matching."""
    missed_notes = []
    correct_notes = []
    extra_played_notes = []

    def onset_match(gt_note, student_note):
        return gt_note[0] == student_note[0] and abs(gt_note[1] - student_note[1]) <= onset_tolerance

    for gt_note in gt_notes:
        if any(onset_match(gt_note, student_note) for student_note in student_notes):
            correct_notes.append(gt_note)
        else:
            missed_notes.append(gt_note)

    for student_note in student_notes:
        if not any(onset_match(student_note, gt_note) for gt_note in gt_notes):
            extra_played_notes.append(student_note)

    return missed_notes, correct_notes, extra_played_notes

def create_midi_from_notes(notes, output_file):
    """Create a MIDI file from specified notes."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes:
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(output_file)

def process_files(dir, score_dir):
    """Process each pair of student and ground truth MIDI files and return new directories."""
    new_extra_dir = []
    new_removed_dir = []
    new_score_dir = []

    for student_midi_file, gt_midi_file in zip(dir, score_dir):
        student_notes = midi_to_note_tuples(student_midi_file)
        gt_notes = midi_to_note_tuples(gt_midi_file)
        missed_notes, correct_notes, extra_played_notes = compare_midi(gt_notes, student_notes)

        # Paths for the new MIDI files
        base_dir = os.path.dirname(student_midi_file)
        base_name = os.path.splitext(os.path.basename(student_midi_file))[0]
        
        missed_path = os.path.join(base_dir, f'{base_name}_missed_notes.mid')
        correct_path = os.path.join(base_dir, f'{base_name}_correct_notes.mid')
        extra_path = os.path.join(base_dir, f'{base_name}_extra_played_notes.mid')

        create_midi_from_notes(missed_notes, missed_path)
        create_midi_from_notes(correct_notes, correct_path)
        create_midi_from_notes(extra_played_notes, extra_path)

        new_extra_dir.append(extra_path)
        new_removed_dir.append(missed_path)
        new_score_dir.append(correct_path)

    return new_extra_dir, new_removed_dir, new_score_dir



# Called by test, TODO: see how to use this.
def evaluate_main(
    dataset_name,  
    test_midi_dir,
    ground_truth,
    enable_instrument_eval=False, # TODO: check what this means
    first_n=None,
):

        
    if dataset_name == "MAESTRO_old" or dataset_name == "Score_Informed":
        # dir = sorted(glob.glob(f"{test_midi_dir}/*/mix.mid"))
        # dir_names = [os.path.basename(os.path.dirname(midi_file)) for midi_file in dir]

        # print("dir:", dir, flush=True)
        # print("dir_names:", dir_names, flush=True)
        
        # print("dir:", dir, flush=True)
        # score_dir = []
        
        
        # for k in range(len(ground_truth)):
        #     for midi_file in dir:
        #         if ground_truth[k]["track_id"] == os.path.basename(os.path.dirname(midi_file)):
        #             print("found", midi_file)
        #             # add the midi file to the list
        #             score_dir.append(ground_truth[k]["midi_path"])

        dir = sorted(glob.glob(f"{test_midi_dir}/*/mix.mid"))
        dir_names = [os.path.basename(os.path.dirname(midi_file)) for midi_file in dir]
        
        # Map track_ids to various midi_paths from ground_truth
        track_to_extra = {gt["track_id"]: gt["extra_notes_midi"] for gt in ground_truth}
        track_to_removed = {gt["track_id"]: gt["removed_notes_midi"] for gt in ground_truth}
        track_to_correct = {gt["track_id"]: gt["correct_notes_midi"] for gt in ground_truth}

        # Build the three directories ordered according to dir
        extra_dir_gt = [track_to_extra[dir_name] for dir_name in dir_names if dir_name in track_to_extra]
        removed_dir_gt = [track_to_removed[dir_name] for dir_name in dir_names if dir_name in track_to_removed]
        correct_dir_gt = [track_to_correct[dir_name] for dir_name in dir_names if dir_name in track_to_correct]


        # Map track_ids to midi_paths from ground_truth
        # used to be score_midi
        track_to_midi = {gt["track_id"]: gt["aligned_midi"] for gt in ground_truth}

        # Build score_dir ordered according to dir
        score_dir = [track_to_midi[dir_name] for dir_name in dir_names if dir_name in track_to_midi]
        extra_dir, removed_dir, correct_dir = process_files(dir, score_dir)
        # Print to verify order and correctness
        print("dir:", dir)
        print("dir_names:", dir_names)
        print("score_dir:", score_dir)
        print("extra_dir:", extra_dir, flush=True)
        print("removed_dir:", removed_dir, flush=True)
        print("correct_dir:", correct_dir, flush=True)
        print("extra_dir_gt:", extra_dir_gt, flush=True)
        print("removed_dir_gt:", removed_dir_gt, flush=True)
        print("correct_dir_gt:", correct_dir_gt, flush=True)
                    
        if first_n:
            dir = dir[:first_n]
            extra_dir = extra_dir[:first_n]
            removed_dir = removed_dir[:first_n]
            correct_dir = correct_dir[:first_n]
            extra_dir_gt = extra_dir_gt[:first_n]
            removed_dir_gt = removed_dir_gt[:first_n]
            correct_dir_gt = correct_dir_gt[:first_n]
            
        fnames = zip(extra_dir, removed_dir, correct_dir, extra_dir_gt, removed_dir_gt, correct_dir_gt)

    elif dataset_name == "CocoChorales":
        pass
    
    else:
        raise ValueError("dataset_name must be either MAESTRO_old, Score_Informed, or CocoChorales_old.")

    def func(item):
        fextra, fremoved, fcorrect, fextra_gt, fremoved_gt, fcorrect_gt = item

        results = {}
        for granularity in ["flat", "full", "midi_class"]:
            # print("\ngranularity:", granularity)
            dic = mt3_program_aware_note_scores(fextra, fremoved, fcorrect, fextra_gt, fremoved_gt, fcorrect_gt, granularity)
            results.update(dic)

        return results

    pbar = tqdm(total=len(dir))
    scores = collections.defaultdict(list)
    # update scores with the results from the evaluation asynchrounously
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start the load operations and mark each future with its URL
        future_to_fname = {executor.submit(func, fname): fname for fname in fnames}
        for future in concurrent.futures.as_completed(future_to_fname):
            try:
                fname = future_to_fname[future]
                dic = future.result()
                for item in dic:
                    scores[item].append(dic[item])
                pbar.update()
            except Exception as e:
                print(str(e))
                traceback.print_exc()
    print(f"individual scores: {scores}", flush=True)

    mean_scores = {k: np.mean(v) for k, v in scores.items() if k != "F1 by program"}
    # maybe we need this. not sure.
    if enable_instrument_eval:
        # get instrument level evaluation
        print("====")
        program_f1_dict = {}
        for item in scores["F1 by program"]:
            for key in item:
                if key not in program_f1_dict:
                    program_f1_dict[key] = []
                program_f1_dict[key].append(item[key])

        d = {
            -1: "Drums",
            0: "Piano",
            1: "Chromatic Percussion",
            2: "Organ",
            3: "Guitar",
            4: "Bass",
            5: "Strings",
            6: "Ensemble",
            7: "Brass",
            8: "Reed",
            9: "Pipe",
            10: "Synth Lead",
            11: "Synth Pad",
            12: "Synth Effects",
        }
        program_f1_dict = {k: np.mean(np.array(v)) for k, v in program_f1_dict.items()}
        for key in d:
            if key == -1:
                print("{}: {:.4}".format(d[key], program_f1_dict[key]))
            elif key * 8 in program_f1_dict:
                print("{}: {:.4}".format(d[key], program_f1_dict[key * 8]))

    return scores, mean_scores



