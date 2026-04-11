# chopper.py
import os
from moviepy import VideoFileClip

def chop_long_video(input_file, output_folder="seed_videos", clip_length=4.0, max_clips=50):
    """Slices one long video into dozens of short training clips."""
    print(f"🎬 Loading {input_file}...")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    video = VideoFileClip(input_file)
    duration = video.duration
    
    clip_count = 0
    
    for start_time in range(0, int(duration), int(clip_length)):
        if clip_count >= max_clips:
            break
            
        end_time = start_time + clip_length
        if end_time > duration:
            break
            
        out_path = os.path.join(output_folder, f"seed_clip_{clip_count:03d}.mp4")
        print(f"✂️ Extracting {out_path}...")
        
        # Cut the clip and save it
        chunk = video.subclipped(start_time, end_time)
        chunk.write_videofile(
            out_path, 
            codec="libx264", 
            audio_codec="aac"
        )
        clip_count += 1

    video.close()
    print(f"\n✅ SUCCESS: Created {clip_count} short training clips in the '{output_folder}' folder!")

if __name__ == "__main__":
    # Put your ONE long video in the main folder and type its name here:
    LONG_VIDEO_NAME = "6.mp4" 
    
    if os.path.exists(LONG_VIDEO_NAME):
        chop_long_video(LONG_VIDEO_NAME)
    else:
        print(f"❌ ERROR: Could not find '{LONG_VIDEO_NAME}'. Make sure it is in the same folder as this script!")