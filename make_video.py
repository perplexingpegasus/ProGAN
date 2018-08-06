from progan_v15 import ProGAN

import librosa
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.editor import AudioFileClip
from sklearn.preprocessing import StandardScaler


def get_z_from_audio(audio, z_length, n_bins=60, hop_length=512, random_state=50):
    np.random.seed(random_state)
    if type(audio) == str:
        audio, sr = librosa.load(audio)

    y = librosa.core.cqt(audio, n_bins=n_bins, hop_length=hop_length)
    mag, phase = librosa.core.magphase(y)
    mag = mag.T
    mag = StandardScaler().fit_transform(mag)

    s0, s1 = mag.shape
    static = np.random.normal(size=[z_length - s1])
    static = np.tile(static, (s0, 1))

    z = np.concatenate((mag, static), 1)
    z = z.T
    np.random.shuffle(z)
    z = z.T
    return z

def make_video(audio, filename, progan, n_bins=60, random_state=0, imgs_per_batch=20):
    y, sr = librosa.load(audio)
    song_length = len(y) / sr
    z_audio = get_z_from_audio(y, z_length=progan.z_length, n_bins=n_bins, random_state=random_state)
    fps = z_audio.shape[0] / song_length
    res = progan.get_cur_res()
    shape = (res, res * 16 // 9, 3)

    imgs = np.zeros(shape=[imgs_per_batch, *shape], dtype=np.float32)

    def make_frame(t):
        global imgs
        cur_frame_idx = int(t * fps)

        if cur_frame_idx >= len(z_audio):
            return np.zeros(shape=shape, dtype=np.uint8)

        if cur_frame_idx % imgs_per_batch == 0:
            imgs = progan.generate(z_audio[cur_frame_idx:cur_frame_idx + imgs_per_batch])
            imgs = imgs[:, :, :res * 8 // 9, :]
            imgs_rev = np.flip(imgs, 2)
            imgs = np.concatenate((imgs, imgs_rev), 2)

        return imgs[cur_frame_idx % imgs_per_batch]

    video_clip = VideoClip(make_frame=make_frame, duration=song_length)
    audio_clip = AudioFileClip(audio)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(filename, fps=fps)

if __name__ == '__main__':
    progan = ProGAN(
        logdir='logdir_v2',
        imgdir='img_arrays',
    )
    make_video('videos\\eco_zones.mp3', 'eco_zones.mp4', progan, random_state=768)