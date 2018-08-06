import os
import numpy as np
from PIL import Image


def generate_square_crops(imgdir, savedir, crops_per_img=10, max_size=1024, filter=Image.BICUBIC):

    img_files = [os.path.join(imgdir, f) for f in os.listdir(imgdir)]
    savedir = os.path.join(savedir, '_temp')
    if not os.path.exists(savedir): os.makedirs(savedir)

    for i, f in enumerate(img_files):

        with Image.open(f) as img:
            width, height = img.size

            if width < max_size or height < max_size: continue

            landscape = width > height
            if landscape:
                new_height = max_size
                new_width = int(width * (max_size / height))
                offset = int(max_size * (width / height - 1) + 1)
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
                offset = int(max_size * (height / width - 1) + 1)

            n_crops = min(offset, crops_per_img)
            window_slide_len = offset / n_crops

            try:
                img = img.convert('RGB')
                img = img.resize((new_width, new_height), filter)

                for j in range(n_crops):
                    shift = int(j * window_slide_len)

                    if landscape: window = (shift, 0, max_size + shift, max_size)
                    else: window = (0, shift, max_size, max_size + shift)

                    cropped_img = img.crop(window)
                    mirror_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)

                    path = os.path.join(savedir, 'img_{}_{}.jpg'.format(i, j))
                    mirror_path = os.path.join(savedir, 'img_{}_{}_mirror.jpg'.format(i, j))
                    cropped_img.save(path, "JPEG")
                    mirror_img.save(mirror_path, "JPEG")

                print('Processed {}\n'.format(f))

            except OSError:
                continue


def resize(savedir, NCHW=True, min_size=4, max_size=1024, max_mem=0.8,
           use_uint8=True, filter=Image.BICUBIC):

    resized_img_dir = os.path.join(savedir, '_temp')
    img_files = [os.path.join(resized_img_dir, f) for f in os.listdir(resized_img_dir)]
    np.random.shuffle(img_files)
    savedir = os.path.join(savedir, 'memmaps')
    if not os.path.exists(savedir): os.makedirs(savedir)

    sizes = [
        2 ** i for i in range(
        int(np.log2(min_size)),
        int(np.log2(max_size)) + 1
    )]

    pixel_bytes = 3 if use_uint8 else 12
    max_bytes = max_mem * 1e9

    for s in sizes:
        max_imgs = int(max_bytes / (pixel_bytes * s ** 2))
        batch_shape = (max_imgs, 3, s, s) if NCHW else (max_imgs, s, s, 3)
        batch = np.zeros(batch_shape, np.uint8)
        img_count = 0
        batch_count = 0

        for f in img_files:

            with Image.open(f) as img:
                width, height = img.size

                if width != s and height != s:
                    img = img.resize((s, s), filter)
                img = np.asarray(img, np.uint8)
                if NCHW:
                    img = np.transpose(img, (2, 0, 1))
                batch[img_count] = img

            if img_count < max_imgs - 1:
                img_count += 1
            else:
                path = os.path.join(savedir, '{}_{}.npy'.format(s, batch_count))
                np.save(path, batch)
                print('Saved {}'.format(path))
                img_count = 0
                batch_count += 1

        if img_count != 0:
            path = os.path.join(savedir, '{}_{}.npy'.format(s, batch_count))
            np.save(path, batch[:img_count])
            print('Saved {}'.format(path))


if __name__ == '__main__':
    imgdir = input('Image directory: ')
    savedir = input('Memmap directory: ')

    #generate_square_crops(imgdir, savedir)
    resize(savedir)