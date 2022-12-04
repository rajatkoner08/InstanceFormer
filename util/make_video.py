import os
import moviepy.video.io.ImageSequenceClip

root='your_root'
dataset = 'analysis'
video = '0fc3e9dbcc' #change it for different videos
image_folder = f'{root}/instanceformer_output/{dataset}/{video}/'
video_name = f'{root}/instanceformer_output/{dataset}/{video}/video.mp4'
print(video_name)
fps=5
image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
image_files.sort(key = lambda x: int(x.split('/')[-1].split('.')[0]))
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(video_name)