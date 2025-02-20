import os
import json
import mir_eval
import glob
import pretty_midi
import numpy as np
import librosa
import note_seq
import collections
import concurrent.futures
import traceback
from tqdm import tqdm
import tempfile


def get_granular_program(program_number, is_drum, granularity_type):
    """
    Returns the granular program number based on the given parameters.

    Parameters:
    program_number (int): The original program number.
    is_drum (bool): Indicates whether the program is a drum program or not.
    granularity_type (str): The type of granularity to apply.

    Returns:
    int: The granular program number.

    """
    if granularity_type == "full":
        return program_number
    elif granularity_type == "midi_class":
        return (program_number // 8) * 8
    elif granularity_type == "flat":
        return 0 if not is_drum else 1


# Standard evaluation Pipeline for MT3
def compute_transcription_metrics(ref_mid, est_mid):
    """Helper function to compute onset/offset, onset only, and frame metrics."""
    ns_ref = note_seq.midi_file_to_note_sequence(ref_mid)
    ns_est = note_seq.midi_file_to_note_sequence(est_mid)
    intervals_ref, pitches_ref, _ = note_seq.sequences_lib.sequence_to_valued_intervals(ns_ref)
    intervals_est, pitches_est, _ = note_seq.sequences_lib.sequence_to_valued_intervals(ns_est)
    
    len_est_intervals = len(intervals_est)
    len_ref_intervals = len(intervals_ref)

    # Handle cases where intervals are empty
    if len_ref_intervals == 0 and len_est_intervals == 0:
        # Both are empty, return perfect scores
        return {
            "len_ref_intervals": len_ref_intervals,
            "len_est_intervals": len_est_intervals,
            "onoff_precision": 1.0,
            "onoff_recall": 1.0,
            "onoff_f1": 1.0,
            "onoff_overlap": 1.0,
            "on_precision": 1.0,
            "on_recall": 1.0,
            "on_f1": 1.0,
            "on_overlap": 1.0,
        }
    elif len_ref_intervals == 0 or len_est_intervals == 0:
        # One of them is empty, return zero scores
        return {
            "len_ref_intervals": len_ref_intervals,
            "len_est_intervals": len_est_intervals,
            "onoff_precision": 0.0,
            "onoff_recall": 0.0,
            "onoff_f1": 0.0,
            "onoff_overlap": 0.0,
            "on_precision": 0.0,
            "on_recall": 0.0,
            "on_f1": 0.0,
            "on_overlap": 0.0,
        }

    # If neither is empty, compute the metrics as usual
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
def mt3_program_aware_note_scores(fextra, fremoved, fmistakes, fname2, granularity_type):
    """
    Edited version of MT3's program aware precision/recall/F1 score.
    We follow Perceiver's evaluation approach which takes only onset and program into account.
    Using MIDIs transcribed from MT3, we managed to get similar results as Perceiver, which is 0.75 for onset F1.
    """
    # ref_extra_mid = pretty_midi.PrettyMIDI(fextra) # reference midi (music score)
    # ref_removed_mid = pretty_midi.PrettyMIDI(fremoved) # reference midi (music score)
    # ref_mistakes_mid = pretty_midi.PrettyMIDI(fmistakes) # reference midi (music score)   
    # List of MIDI files to combine
    midi_files = [fextra, fremoved, fmistakes]

    # Create a new PrettyMIDI object for the combined MIDI
    combined_midi = pretty_midi.PrettyMIDI()

    # Load each file and add its instruments to the combined MIDI
    for midi_file in midi_files:
        # Load the current MIDI file
        current_midi = pretty_midi.PrettyMIDI(midi_file)
        
        # Add each instrument from the current file to the combined MIDI file
        for instrument in current_midi.instruments:
            combined_midi.instruments.append(instrument)   
    ref_mid = combined_midi # reference midi (music score) 
    est_mid = pretty_midi.PrettyMIDI(fname2) # estimated midi (transcription)
    # Ensure est_mid has exactly three instruments
    while len(est_mid.instruments) < 3:
        # Create an empty instrument
        empty_instrument = pretty_midi.Instrument(program=0)  # program=0 is usually Acoustic Grand Piano, can be changed
        # Add the empty instrument to the beginning
        est_mid.instruments.insert(0, empty_instrument)
    # Print details about each instrument (treated as a track)
    for index, instrument in enumerate(ref_mid.instruments):
        is_drum = 'Drum' if instrument.is_drum else 'Not Drum'
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print(f"Track {index}: Instrument {instrument.program}, Name: {instrument_name}, {is_drum}")
        
    for index, instrument in enumerate(est_mid.instruments):
        is_drum = 'Drum' if instrument.is_drum else 'Not Drum'
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        print(f"Track {index}: Instrument {instrument.program}, Name: {instrument_name}, {is_drum}")
        
    # Write the combined MIDI to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_midi_file:
        combined_midi.write(temp_midi_file.name)
        temp_midi_file_path = temp_midi_file.name

    res = dict() # results
    ref_ns = note_seq.midi_file_to_note_sequence(temp_midi_file_path)
    
    est_ns = note_seq.midi_file_to_note_sequence(fname2)
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

        

        def convert_instrument(ns, target_instrument=0):
            for note in ns.notes:
                note.instrument = target_instrument
            return ns

        # Convert the instrument in the estimated sequence to match the reference sequence (convert both to grand Piano)
        est_ns = convert_instrument(est_ns)
        ref_ns = convert_instrument(ref_ns)

        est_intervals, est_pitches, est_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
        )
        # intervals is like (note.start_time, note.end_time)

        ref_intervals, ref_pitches, ref_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)
        )

        # Precision / recall / F1 using onsets (and pitches) only.
        # looks like we can just do this seperately for each type of error!
        if len(est_intervals) == 0 or len(ref_intervals) == 0:
            precision, recall, f_measure = 0.0, 0.0, 0.0
        elif len(est_intervals) == 0 and len(ref_intervals) == 0:
            precision, recall, f_measure = 1.0, 1.0, 1.0
        else:  
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
        print(f"precision={precision} recall={recall} f_measure={f_measure}")
        
    # Function to convert instrument program to name
    def get_instrument_name(program):
        return pretty_midi.program_to_instrument_name(program)
        
    # Iterate over each estimated track and corresponding reference track
    if len(est_mid.instruments) != len(midi_files):
        print(f"Number of instruments in the estimated MIDI ({len(est_mid.instruments)}) does not match the number of MIDI files ({len(midi_files)})")   
        # assign 0 to all the scores
        precision = 0
        recall = 0
        f_measure = 0
        for track_index in range(len(midi_files)):
            
            # Initialize nested dictionaries if they do not exist
            if f"Track {track_index} precision" not in res:
                res[f"Track {track_index} precision"] = {}
            if f"Track {track_index} recall" not in res:
                res[f"Track {track_index} recall"] = {}
            if f"Track {track_index} F1" not in res:
                res[f"Track {track_index} F1"] = {}

            # Store results for this track
            instrument_name = get_instrument_name(program)
            res[f"Track {track_index} precision"].setdefault(instrument_name, []).append(precision)
            res[f"Track {track_index} recall"].setdefault(instrument_name, []).append(recall)
            res[f"Track {track_index} F1"].setdefault(instrument_name, []).append(f_measure)
        
        
        program_f1 = {}
        program_precision = {}
        program_recall = {}
        
        ##########################
        res.update(
            {
                f"Onset + program precision ({granularity_type})": precision,
                f"Onset + program recall ({granularity_type})": recall,
                f"Onset + program F1 ({granularity_type})": f_measure,
                # f'Drum onset precision ({granularity_type})': drum_precision,
                # f'Drum onset recall ({granularity_type})': drum_recall,
                # f'Drum onset F1 ({granularity_type})': drum_f_measure,
                # f'Nondrum onset + program precision ({granularity_type})':
                #     nondrum_precision,
                # f'Nondrum onset + program recall ({granularity_type})':
                #     nondrum_recall,
                # f'Nondrum onset + program F1 ({granularity_type})':
                #     nondrum_f_measure
                f"F1 by program({granularity_type})": program_f1,
                f"Precision by program({granularity_type})": program_precision,
                f"Recall by program({granularity_type})": program_recall,
            }
        )
        
        return res
    else:
        # Iterate over each estimated track and corresponding reference track
        for track_index, (ref_mid_file, est_instrument) in enumerate(zip(midi_files, est_mid.instruments)):
            # Skip if estimated track has more instruments than the reference tracks
            if not os.path.exists(ref_mid_file):
                print(f"File does not exist: {ref_mid_file}")
                continue  # Skip this iteration if the file does not exist

            # Load the reference MIDI file
            ref_mid = pretty_midi.PrettyMIDI(ref_mid_file)
            if len(ref_mid.instruments) == 0:
                print(f"No instruments found in reference MIDI file: {ref_mid_file}")
                continue
            ref_instrument = ref_mid.instruments[0]
            program = ref_instrument.program

            
            ref_ns = note_seq.midi_file_to_note_sequence(ref_mid_file)

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
            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                precision, recall, f_measure = 0.0, 0.0, 0.0
            elif len(est_intervals) == 0 and len(ref_intervals) == 0:
                precision, recall, f_measure = 1.0, 1.0, 1.0
            else:  
                precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals=ref_intervals,
                    ref_pitches=ref_pitches,
                    est_intervals=est_intervals,
                    est_pitches=est_pitches,
                    offset_ratio=None
                )

            # Initialize nested dictionaries if they do not exist
            if f"Track {track_index} precision" not in res:
                res[f"Track {track_index} precision"] = {}
            if f"Track {track_index} recall" not in res:
                res[f"Track {track_index} recall"] = {}
            if f"Track {track_index} F1" not in res:
                res[f"Track {track_index} F1"] = {}

            # Store results for this track
            instrument_name = get_instrument_name(program)
            res[f"Track {track_index} precision"].setdefault(instrument_name, []).append(precision)
            res[f"Track {track_index} recall"].setdefault(instrument_name, []).append(recall)
            res[f"Track {track_index} F1"].setdefault(instrument_name, []).append(f_measure)

            print(f"Track {track_index}, Instrument: {instrument_name}, Program: {program}")
            print(f"precision={precision} recall={recall} f_measure={f_measure}")
        
            
        # Print the resulting res dictionary for debugging
        print(f"debug res: {res}", flush=True)

        # group notes by program number
        ref_inst_to_notes_mapping = {}
        est_inst_to_notes_mapping = {}

        # this part calculates multi-instrument onset F1 score
        # based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L36

        # following MT3, this will group notes under the same instrument program, determined by the granularity
        for inst in ref_mid.instruments:
            cur_ref_program = get_granular_program(
                inst.program, inst.is_drum, granularity_type
            )
            if (cur_ref_program, inst.is_drum) in ref_inst_to_notes_mapping:
                ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] += [
                    note for note in inst.notes
                ]
            else:
                ref_inst_to_notes_mapping[(cur_ref_program, inst.is_drum)] = [
                    note for note in inst.notes
                ]
        # In order to calculate the F1 score, we need to match the instruments in the reference and estimated MIDI files
        for inst_est, inst_ref in zip(est_mid.instruments, ref_mid.instruments):
            est_program = get_granular_program(
                inst_ref.program, inst_ref.is_drum, granularity_type
            )
            if (est_program, inst_ref.is_drum) in est_inst_to_notes_mapping:
                est_inst_to_notes_mapping[(est_program, inst_ref.is_drum)] += [
                    note for note in inst_est.notes
                ]   
            else:
                est_inst_to_notes_mapping[(est_program, inst_ref.is_drum)] = [
                    note for note in inst_est.notes
                ]
                

        # this part is based on: https://github.com/magenta/mt3/blob/main/mt3/metrics.py#L82
        program_and_is_drum_tuples = set(ref_inst_to_notes_mapping.keys()) | set(
            est_inst_to_notes_mapping.keys()
        )

        drum_precision_sum = 0.0
        drum_precision_count = 0
        drum_recall_sum = 0.0
        drum_recall_count = 0

        nondrum_precision_sum = 0.0
        nondrum_precision_count = 0
        nondrum_recall_sum = 0.0
        nondrum_recall_count = 0

        program_f1 = {}
        program_precision = {}
        program_recall = {}
        print(f"program and is drum tuples: ", program_and_is_drum_tuples)
        for program, is_drum in program_and_is_drum_tuples:
            if (program, is_drum) in ref_inst_to_notes_mapping:
                ref_notes = ref_inst_to_notes_mapping[(program, is_drum)]
                ref_intervals = np.array([[note.start, note.end] for note in ref_notes])
                ref_pitches = np.array(
                    [librosa.midi_to_hz(note.pitch) for note in ref_notes]
                )
            else:
                # ref does not have this instrument
                ref_intervals = np.zeros((0, 2))
                ref_pitches = np.zeros(0)

            if (program, is_drum) in est_inst_to_notes_mapping:
                est_notes = est_inst_to_notes_mapping[(program, is_drum)]
                est_intervals = np.array([[note.start, note.end] for note in est_notes])
                est_pitches = np.array(
                    [librosa.midi_to_hz(note.pitch) for note in est_notes]
                )
            else:
                print(f"est does not have this instrument: program={program} is_drum={is_drum}")
                # est does not have this instrument
                est_intervals = np.zeros((0, 2))
                est_pitches = np.zeros(0)

            # NOTE: like Perceiver, disable offset calculation
            args = {
                "ref_intervals": ref_intervals,
                "ref_pitches": ref_pitches,
                "est_intervals": est_intervals,
                "est_pitches": est_pitches,
                "offset_ratio": None,
            }
            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                precision, recall, f_measure = 0.0, 0.0, 0.0
            elif len(est_intervals) == 0 and len(ref_intervals) == 0:
                precision, recall, f_measure = 1.0, 1.0, 1.0
            else:  
                # f_measure is the F1 score
                precision, recall, f_measure, unused_avg_overlap_ratio = (
                    mir_eval.transcription.precision_recall_f1_overlap(**args)
                )

            # print(f"program={program} is_drum={is_drum} est={est_pitches.shape[0]} ref={ref_pitches.shape[0]}")
            # print(f"precision={precision} recall={recall} f_measure={f_measure}")
            # print(f"est_intervals={len(est_intervals)} ref_intervals={len(ref_intervals)}")
            # print("======")

            # if granularity_type == "midi_class":
            #     if is_drum:
            #         program_f1[-1] = f_measure
            #     else:
            #         print(f"program={program} precision={precision} recall={recall} f_measure={f_measure}", flush=True)
            #         program_f1[program] = f_measure
            if granularity_type == "full":
                print(f" assigning to program_f1: program={program} precision={precision} recall={recall} f_measure={f_measure}", flush=True)
                program_f1[program] = f_measure
                program_precision[program] = precision
                program_recall[program] = recall

            if is_drum:
                drum_precision_sum += precision * len(est_intervals)
                drum_precision_count += len(est_intervals)
                drum_recall_sum += recall * len(ref_intervals)
                drum_recall_count += len(ref_intervals)
            else:
                nondrum_precision_sum += precision * len(est_intervals)
                nondrum_precision_count += len(est_intervals)
                nondrum_recall_sum += recall * len(ref_intervals)
                nondrum_recall_count += len(ref_intervals)

        precision_sum = drum_precision_sum + nondrum_precision_sum
        precision_count = drum_precision_count + nondrum_precision_count
        recall_sum = drum_recall_sum + nondrum_recall_sum
        recall_count = drum_recall_count + nondrum_recall_count

        # print(f"precision_sum={precision_sum} precision_count={precision_count}")
        # print(f"recall_sum={recall_sum} recall_count={recall_count}")

        precision = (precision_sum / precision_count) if precision_count else 0
        recall = (recall_sum / recall_count) if recall_count else 0
        f_measure = mir_eval.util.f_measure(precision, recall)
        print(f"precision={precision} recall={recall} f_measure={f_measure}", flush=True)

        drum_precision = (
            (drum_precision_sum / drum_precision_count) if drum_precision_count else 0
        )
        drum_recall = (drum_recall_sum / drum_recall_count) if drum_recall_count else 0
        drum_f_measure = mir_eval.util.f_measure(drum_precision, drum_recall)

        nondrum_precision = (
            (nondrum_precision_sum / nondrum_precision_count)
            if nondrum_precision_count
            else 0
        )
        nondrum_recall = (
            (nondrum_recall_sum / nondrum_recall_count) if nondrum_recall_count else 0
        )
        nondrum_f_measure = mir_eval.util.f_measure(nondrum_precision, nondrum_recall)

        res.update(
            {
                f"Onset + program precision ({granularity_type})": precision,
                f"Onset + program recall ({granularity_type})": recall,
                f"Onset + program F1 ({granularity_type})": f_measure,
                # f'Drum onset precision ({granularity_type})': drum_precision,
                # f'Drum onset recall ({granularity_type})': drum_recall,
                # f'Drum onset F1 ({granularity_type})': drum_f_measure,
                # f'Nondrum onset + program precision ({granularity_type})':
                #     nondrum_precision,
                # f'Nondrum onset + program recall ({granularity_type})':
                #     nondrum_recall,
                # f'Nondrum onset + program F1 ({granularity_type})':
                #     nondrum_f_measure
                f"F1 by program({granularity_type})": program_f1,
                f"Precision by program({granularity_type})": program_precision,
                f"Recall by program({granularity_type})": program_recall,
            }
        )
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

# Called by test, TODO: see how to use this.

def evaluate_main(
    dataset_name,  # "MAESTRO", "Score_Informed", or "CocoChorales"
    test_midi_dir,
    output_json_file,  # Path to the JSON file with pre-cached file paths
    ground_truth,
    enable_instrument_eval=True, # TODO: check what this means
    first_n=None,
):
    if dataset_name in ["MAESTRO", "Score_Informed", "CocoChorales"]:
        # Assuming test_midi_dir and output_json_file are defined
        est = sorted(glob.glob(f"{test_midi_dir}/*/*.mid"))

        # Create a dictionary mapping track ID to a list of filenames, ensuring all are in lowercase and spaces are replaced with underscores
        est_files = {}
        for midi_file in est:
            track_id = os.path.basename(os.path.dirname(midi_file))
            filename = os.path.basename(midi_file).lower().replace(' ', '_')
            if track_id in est_files:
                est_files[track_id].append(filename)
            else:
                est_files[track_id] = [filename]

        # Ensure sub-files are sorted within each track
        for track_id in est_files:
            est_files[track_id].sort()

        # Debugging: Print the contents of est_files
        print(f"Contents of est_files: {json.dumps(est_files, indent=2)}", flush=True)

        # Load pre-cached paths from JSON
        with open(output_json_file, 'r') as json_file:
            files_dict = json.load(json_file)

        # Function to convert filenames from lowercase_with_underscores to Capitalized With Spaces
        def convert_filename(filename):
            first_part, rest = filename.split('_', 1)
            rest_parts = ' '.join(word.capitalize() for word in rest.split('_'))
            return f"{first_part}_{rest_parts}"

        # Define a function to filter paths based on filenames from est_files, safely handling multiple files per track ID
        def filter_paths(track_id, paths):
            required_files = est_files.get(track_id, [])  # Get the list of filenames, defaulting to empty if not found
            # Debugging: Print the track ID and required files
            print(f"Track ID: {track_id}, Required Files: {required_files}", flush=True)
            filtered_paths = []
            for path in paths:
                original_filename = os.path.basename(path)
                converted_filename = original_filename.lower().replace(' ', '_')
                if converted_filename in required_files:
                    filtered_paths.append(path)
                    # Debugging: Print the original and converted path
                    print(f"Original: {original_filename} -> Converted: {converted_filename}", flush=True)
            return filtered_paths

        # Prepare mappings from the JSON file using the track IDs in ground_truth
        track_to_extra = {gt["track_id"]: sorted(filter_paths(gt["track_id"], files_dict["extra_notes"][gt["track_id"]])) for gt in ground_truth if gt["track_id"] in files_dict["extra_notes"]}
        track_to_removed = {gt["track_id"]: sorted(filter_paths(gt["track_id"], files_dict["removed_notes"][gt["track_id"]])) for gt in ground_truth if gt["track_id"] in files_dict["removed_notes"]}
        track_to_mistake = {gt["track_id"]: sorted(filter_paths(gt["track_id"], files_dict["correct_notes"][gt["track_id"]])) for gt in ground_truth if gt["track_id"] in files_dict["correct_notes"]}

        # Collect the directories ordered according to track IDs in est_files and ensure file path case insensitivity
        extra_dir = [path for track_id in est_files for path in track_to_extra.get(track_id, [])]
        removed_dir = [path for track_id in est_files for path in track_to_removed.get(track_id, [])]
        mistake_dir = [path for track_id in est_files for path in track_to_mistake.get(track_id, [])]

        # Debugging: Print the directories to debug
        print(f'Est: {est}', flush=True)
        print(f'Extra Dir: {extra_dir}', flush=True)
        print(f'Removed Dir: {removed_dir}', flush=True)
        print(f'Mistake Dir: {mistake_dir}', flush=True)

        if first_n:
            est = est[:first_n]
            extra_dir = extra_dir[:first_n]
            removed_dir = removed_dir[:first_n]
            mistake_dir = mistake_dir[:first_n]
            
        fnames = zip(extra_dir, removed_dir, mistake_dir, est)
    
    else:
        raise ValueError("dataset_name must be either MAESTRO, Score_Informed, or CocoChorales.")


# Add your function calls here to test

    def func(item):
        fextra, fremoved, fmistake, fname2 = item

        results = {}
        for granularity in ["flat", "full", "midi_class"]:
            # print("\ngranularity:", granularity)
            dic = mt3_program_aware_note_scores(fextra, fremoved, fmistake, fname2, granularity)
            results.update(dic)

        return results

    pbar = tqdm(total=len(extra_dir))
    scores = collections.defaultdict(list)

    # Update scores with the results from the evaluation asynchronously
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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

    print(f"scores: {scores}")
    
    # Define the pattern to exclude keys
    exclude_patterns = ["F1 by program", "Precision by program", "Recall by program", "Track"]

    # Function to check if a key matches any of the exclude patterns
    def should_exclude(key):
        key_str = str(key)  # Ensure the key is a string
        return any(pattern in key_str for pattern in exclude_patterns)
    
    mean_scores = {k: np.mean(v) for k, v in scores.items() if not should_exclude(k)}

    # Additional code to calculate track-specific, instrument-specific scores
    track_instrument_scores = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    # Fill track_instrument_scores with values from scores
    for key, value_list in scores.items():
        if "Track" in str(key):  # Ensure the key is treated as a string
            parts = key.split(" ")
            track_index = int(parts[1])
            for program_dict in value_list:
                for program, metrics in program_dict.items():
                    instrument_name = program
                    if isinstance(metrics, dict):
                        for metric, score_list in metrics.items():
                            # Look here track_index... XXXX
                            track_instrument_scores[track_index][instrument_name][metric].extend(score_list)
                    elif isinstance(metrics, list):
                        for score in metrics:
                            # Look here track_index... XXXX
                            track_instrument_scores[track_index][instrument_name][key.split()[-1]].append(score)
                    else:
                        print(f"Unexpected metrics type for program {program} in track {track_index}: {type(metrics)}")

    print(f"Track and instrument scores: {track_instrument_scores}", flush=True)

    # Calculate mean scores for each track and instrument
    mean_track_instrument_scores = {
        track_index: {
            instrument: {
                metric: np.mean(scores) for metric, scores in metrics.items()
            }
            for instrument, metrics in instruments.items()
        }
        for track_index, instruments in track_instrument_scores.items()
    }

    print(f"Mean track and instrument scores: {mean_track_instrument_scores}", flush=True)

    if enable_instrument_eval:
        program_f1_dict = {}
        program_precision_dict = {}
        program_recall_dict = {}
        print("scores[F1 by program(full)]:", scores["F1 by program(full)"], flush=True)
        for item in scores["F1 by program(full)"]:
            for key in item:
                if key not in program_f1_dict:
                    program_f1_dict[key] = []
                program_f1_dict[key].append(item[key])
        for item in scores["Precision by program(full)"]:
            for key in item:
                if key not in program_precision_dict:
                    program_precision_dict[key] = []
                program_precision_dict[key].append(item[key])
        for item in scores["Recall by program(full)"]:
            for key in item:
                if key not in program_recall_dict:
                    program_recall_dict[key] = []
                program_recall_dict[key].append(item[key])
                
        # TODO: This is empty
        print("program_f1_dict:", program_f1_dict, flush=True)
        
        # d = {
        #     -1: "Drums",
        #     0: "Piano",
        #     1: "Chromatic Percussion",
        #     2: "Organ",
        #     3: "Guitar",
        #     4: "Bass",
        #     5: "Strings",
        #     6: "Ensemble",
        #     7: "Brass",
        #     8: "Reed",
        #     9: "Pipe",
        #     10: "Synth Lead",
        #     11: "Synth Pad",
        #     12: "Synth Effects",
        # }
        program_f1_dict = {k: np.mean(np.array(v)) for k, v in program_f1_dict.items()}
        program_precision_dict = {k: np.mean(np.array(v)) for k, v in program_precision_dict.items()}
        program_recall_dict = {k: np.mean(np.array(v)) for k, v in program_recall_dict.items()}
        # KEY IS CURRENTLY ALL ZERO
        # loop over all instruments
        for key in range(0, 128):
            if key in program_f1_dict:
                print("{} F1: {:.4}".format(pretty_midi.program_to_instrument_name(key), program_f1_dict[key]))
            if key in program_precision_dict:
                print("{} Precision: {:.4}".format(pretty_midi.program_to_instrument_name(key), program_precision_dict[key]))
            if key in program_recall_dict:
                print("{} Recall: {:.4}".format(pretty_midi.program_to_instrument_name(key), program_recall_dict[key]))
            # elif key * 8 in program_f1_dict:
            #     print("{}: {:.4}".format(d[key], program_f1_dict[key * 8]))


    return mean_scores, mean_track_instrument_scores, track_instrument_scores
