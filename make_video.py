import librosa
import numpy as np
from progan_v15 import ProGAN
from moviepy.video.VideoClip import VideoClip
from moviepy.editor import AudioFileClip


def get_z_from_audio(audio, z_length, n_bins=84, hop_length=512, random_state=50):
    np.random.seed(random_state)
    if type(audio) == str:
        audio, sr = librosa.load(audio)

    y = librosa.core.cqt(audio, n_bins=n_bins, hop_length=hop_length)
    y, phase = librosa.core.magphase(y)
    y = (y - np.mean(y)) / np.std(y)

    s0, s1 = y.shape
    static = np.random.normal(size=[z_length - s0])
    static = np.tile(static, (s1, 1))
    static = static.T

    z = np.concatenate((y, static), 0)
    np.random.shuffle(z)
    z = z.T
    return z

def make_video(audio, filename, progan, n_bins=84, random_state=0):
    y, sr = librosa.load(audio)
    song_length = len(y) / sr
    z_audio = get_z_from_audio(y, z_length=progan.z_length, n_bins=n_bins, random_state=random_state)
    fps = z_audio.shape[0] / song_length
    shape = progan.generate(z_audio[0]).shape
    def make_frame(t):
        cur_frame_idx = int(t * fps)
        if cur_frame_idx < len(z_audio):
            img = progan.generate(z_audio[cur_frame_idx])
        else:
            img = np.zeros(shape=shape, dtype=np.uint8)
        return img

    video_clip = VideoClip(make_frame=make_frame, duration=song_length)
    audio_clip = AudioFileClip(audio)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(filename, fps=fps)

if __name__ == '__main__':
    progan = ProGAN(
        logdir='logdir_v2',
        img_dir='img_arrays',
    )
    for i in range(10):
        make_video('videos\\natural.mp3', 'natural{}.mp4'.format(i), progan, random_state=i)