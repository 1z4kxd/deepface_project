import argparse
import sys

import partner_a_vision
import partner_b_ai

def process_webcam_frame(frame, faces):
    """
    The bridge function: Takes the raw frame and detected faces from Partner A,
    and passes them to Partner B's function to draw the rich graphics.
    """
    processed_frame = partner_b_ai.draw_rich_visuals(frame, faces)
    return processed_frame

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition System")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['batch', 'webcam'],
        help="Execution mode: 'batch' for static image analysis or 'webcam' for live feed."
    )
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        print("Initializing Batch Mode...")
        partner_b_ai.run_batch_analysis(dataset_path="dataset")
        
    elif args.mode == 'webcam':
        print("Initializing Webcam Mode...")
        partner_a_vision.run_webcam_loop(process_frame_callback=process_webcam_frame)
        
    else:
        print("Error: Invalid mode selected.")
        sys.exit(1)

if __name__ == "__main__":
    main()