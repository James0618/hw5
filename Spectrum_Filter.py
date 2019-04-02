import numpy as np
import cv2
import re


r = r'\.[a-zA-Z0-9]+'


class Filter:
    def __init__(self, method, task):
        self.method = method
        self.task = task

    def filtering(self, image_name, n, d0):
        image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
        image = cv2.imread(image_path, 0)
        image = image
        global r
        image_name = re.sub(r, '', image_name)
        height, width = image.shape
        frequency = np.fft.fft2(image)
        transformed = np.fft.fftshift(frequency)
        power1 = sum(np.abs(sum(transformed ** 2)))
        for i in range(height):
            for j in range(width):
                temp_distance = np.sqrt((i - height / 2) ** 2 + (j - width / 2) ** 2)
                H = self.cal_h(temp_distance=temp_distance, n=n, d0=d0, height=height, width=width)
                transformed[i][j] = transformed[i][j] * H

        power2 = sum(np.abs(sum(transformed ** 2)))
        freq_image = 20 * np.log(np.abs(transformed) + 1)
        filted_image = np.abs(np.fft.ifft2(transformed))
        # save the image
        if self.method == 'butterworth_lowpast':
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}_{}_{}.jpg'.
                        format(self.task, image_name, self.method, n, d0), filted_image)
        elif self.method == 'gaussian_lowpast':
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}_{}.jpg'.
                        format(self.task, image_name, self.method, d0), filted_image)
        elif self.method == 'butterworth_highpast':
            filted_image = filted_image + image
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}_{}_{}.jpg'.
                        format(self.task, image_name, self.method, n, d0), filted_image)
        elif self.method == 'gaussian_highpast':
            filted_image = filted_image + image
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}_{}.jpg'.
                        format(self.task, image_name, self.method, d0), filted_image)
        elif self.method == 'laplace_highpast':
            filted_image = filted_image + image
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}.jpg'.
                        format(self.task, image_name, self.method), filted_image)
        elif self.method == 'unmask_highpast':
            filted_image = filted_image + image
            cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}.jpg'.
                        format(self.task, image_name, self.method), filted_image)
        return power2 / power1

    def cal_h(self, temp_distance, n, d0, height=256, width=256):
        if self.method == 'butterworth_lowpast':
            return 1/(1 + (temp_distance/d0)**(2*n))
        elif self.method == 'gaussian_lowpast':
            return np.exp(-((temp_distance/d0)**2)/2)
        elif self.method == 'butterworth_highpast':
            return 1/(1 + (d0/temp_distance)**(2*n))
        elif self.method == 'gaussian_highpast':
            return 1 - np.exp(-((temp_distance/d0)**2)/2)
        elif self.method == 'laplace_highpast':
            return 4*temp_distance**2/(height**2+width**2)
        elif self.method == 'unmask_highpast':
            return 0.5 + 0.75*(1 - np.exp(-((temp_distance/50)**2)/2))
        else:
            return 0


def task1():
    print('####################### task1 #######################')
    # butterworth lowpast filter
    Butterworth_lowpast = Filter('butterworth_lowpast', 1)
    d0_list = [10, 20, 40, 100]
    for i in range(1, 4):
        for j in range(4):
            spectral_ratio_1 = Butterworth_lowpast.filtering(image_name='test1.pgm', n=i, d0=d0_list[j])
            print("1: n={}, d0={}, spetral ratio={:.2f}%".format(i, d0_list[j], 100*spectral_ratio_1))

    for i in range(1, 4):
        for j in range(4):
            spectral_ratio_2 = Butterworth_lowpast.filtering(image_name='test2.tif', n=i, d0=d0_list[j])
            print("2: n={}, d0={}, spetral ratio={:.2f}%".format(i, d0_list[j], 100*spectral_ratio_2))

    # gaussian lowpast filter
    d0_list = [10, 20, 40, 100]
    Gaussian_lowpast = Filter('gaussian_lowpast', 1)
    for i in range(4):
        spectral_ratio_1 = Gaussian_lowpast.filtering(image_name='test1.pgm', n=i, d0=d0_list[i])
        print("1: d0={}, spetral ratio={:.2f}%".format(d0_list[i], 100*spectral_ratio_1))

    for i in range(4):
        spectral_ratio_2 = Gaussian_lowpast.filtering(image_name='test2.tif', n=i, d0=d0_list[i])
        print("2: d0={}, spetral ratio={:.2f}%".format(d0_list[i], 100*spectral_ratio_2))


def task2():
    print('####################### task2 #######################')
    # butterworth highpast filter
    Butterworth_lowpast = Filter('butterworth_highpast', 2)
    d0_list = [10, 20, 40, 100]
    for i in range(1, 4):
        for j in range(4):
            spectral_ratio_1 = Butterworth_lowpast.filtering(image_name='test3.pgm', n=i, d0=d0_list[j])
            print("1: n={}, d0={}, spetral ratio={:.2f}%".format(i, d0_list[j], 100 * spectral_ratio_1))

    for i in range(1, 4):
        for j in range(4):
            spectral_ratio_2 = Butterworth_lowpast.filtering(image_name='test4.tif', n=i, d0=d0_list[j])
            print("2: n={}, d0={}, spetral ratio={:.2f}%".format(i, d0_list[j], 100 * spectral_ratio_2))

    # gaussian highpast filter
    d0_list = [10, 20, 40, 100]
    Gaussian_lowpast = Filter('gaussian_highpast', 2)
    for i in range(4):
        spectral_ratio_1 = Gaussian_lowpast.filtering(image_name='test3.pgm', n=i, d0=d0_list[i])
        print("1: d0={}, spetral ratio={:.2f}%".format(d0_list[i], 100 * spectral_ratio_1))

    for i in range(4):
        spectral_ratio_2 = Gaussian_lowpast.filtering(image_name='test4.tif', n=i, d0=d0_list[i])
        print("2: d0={}, spetral ratio={:.2f}%".format(d0_list[i], 100 * spectral_ratio_2))


def task3():
    print('####################### task3 #######################')
    # butterworth highpast filter
    Laplace_highpast = Filter('laplace_highpast', 3)
    _ = Laplace_highpast.filtering(image_name='test3.pgm', n=0, d0=0)
    _ = Laplace_highpast.filtering(image_name='test4.tif', n=0, d0=0)

    Unmask_highpast = Filter('unmask_highpast', 3)
    _ = Unmask_highpast.filtering(image_name='test3.pgm', n=0, d0=0)
    _ = Unmask_highpast.filtering(image_name='test4.tif', n=0, d0=0)


if __name__ == '__main__':
    # task1()
    # task2()
    task3()
