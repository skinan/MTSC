
import json

import os
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset



def fft_find_each_amplitude(data, target_period):
    '''
    For each element in a batch
    :param data:
    :param target_period:
    :return: target amplitudes
    '''
    batch_size = data.shape[0]
    sequence_length = data.shape[1]
    T = 1.0 / sequence_length  # sampling interval

    # Initialize an array to store the amplitude for each batch element
    amplitudes = torch.zeros((batch_size, 1))

    for i in range(batch_size):
        # For each batch element, average the num_channels dimension
        averaged_data = data[i].mean(axis=-1)

        # Compute FFT
        yf = np.fft.fft(averaged_data)
        xf = np.fft.fftfreq(sequence_length, T)[:sequence_length // 2]
        power_spectrum = 2.0 / sequence_length * np.abs(yf[:sequence_length // 2])

        # Calculate the target frequency
        target_frequency = sequence_length / target_period
        # Find the closest frequency index
        closest_index = np.argmin(np.abs(xf - target_frequency))
        closest_amplitude = power_spectrum[closest_index]

        # Store the amplitude of the current batch element
        amplitudes[i] = closest_amplitude

    return amplitudes


if __name__ == '__main__':
    device = torch.device('cpu')
    dataset_path = './dataset/General/'
    dataset_name_list = [
        "EthanolConcentration",
        "FaceDetection",
        "Handwriting",
        "Heartbeat",
        "JapaneseVowels",
        "PEMS-SF",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "UWaveGestureLibrary",
    ]
    for dataset_name in dataset_name_list:
        X_train, y_train, X_test, y_test = TSC_multivariate_data_loader(dataset_path, dataset_name)
        print('[INFO] running at:', dataset_name)
        # load multivariate data
        print('train data shape', X_train.shape)

        if X_train.shape[-1] != X_test.shape[-1]:
            print('[INFO]: seq length between train and test unmatched')
            target_length = max(X_train.shape[-1], X_test.shape[-1])
            if X_train.shape[-1] > X_test.shape[-1]:
                X_test = fill_out_with_Nan(X_test, target_length)
            else:
                X_train = fill_out_with_Nan(X_train, target_length)

        print('train data shape', X_train.shape)

        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()

        # replace NaN with 0
        X_train[torch.isnan(X_train)] = 0
        X_test[torch.isnan(X_test)] = 0

        # covert numpy to pytorch tensor and put into gpu
        X_train.requires_grad = False
        if len(X_train.shape) == 3:
            X_train = X_train.to(device)
        else:
            X_train = X_train.unsqueeze_(1).to(device)
        y_train = torch.LongTensor(y_train).to(device)

        X_test.requires_grad = False
        if len(X_test.shape) == 3:
            X_test = X_test.to(device)
        else:
            X_test = X_test.unsqueeze_(1).to(device)
        y_test = torch.LongTensor(y_test).to(device)

        X_train_fft = X_train.permute(0, 2, 1)
        fft_main_periods_wo_duplicates(X_train_fft, 5, dataset_name)
    #     # periods, amplitudes = FFT_for_Period(X_train_fft, k=5)
    #     # print("X_train FFT periods:", periods)
    #
    #     train_dataset = TensorDataset(X_train, y_train)
    #     train_loader = DataLoader(train_dataset, batch_size=5,
    #                               shuffle=True)
    #     test_dataset = TensorDataset(X_test, y_test)
    #     test_loader = DataLoader(test_dataset, batch_size=5,
    #                              shuffle=False)

        # i = 0
        # for sample in train_loader:
        #     i += 1
        #     x = sample[0]
        #     print(x.shape)
        #     x_fft = x.permute(0, 2, 1)  # (batch_size, seq_length, num_channels)
        #     fft_main_periods(x_fft, 10)
        #     fft_find_amplitude(x_fft, 350)
        #     # w_periods, w_amplitudes = DWT_for_Period(x_fft, k=5)
        #     # print("DWT periods:", w_periods)
        #     if i == 3:
        #         break

    # import psutil
    #
    # # 查看CPU使用率
    # cpu_usage = psutil.cpu_percent(interval=1)
    # print(f"CPU使用率: {cpu_usage}%")
    #
    # # 查看内存使用情况
    # memory_info = psutil.virtual_memory()
    # print(f"总内存: {memory_info.total / (1024 ** 3):.2f} GB")
    # print(f"已使用内存: {memory_info.used / (1024 ** 3):.2f} GB")
    # print(f"可用内存: {memory_info.available / (1024 ** 3):.2f} GB")
    # print(f"内存使用率: {memory_info.percent}%")
    #
    # # 查看每个进程的内存和CPU使用情况
    # for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
    #     print(proc.info)

    # Plot a subset of the training data
    # plot_time_series(X_train, num_series=3, series_length=X_train.shape[2])

    # train_file_path = './dataset/General/JapaneseVowels/JapaneseVowels_TRAIN.ts'
    # # train_file_path = './dataset/NATOPS/NATOPS_TRAIN.ts'
    # test_file_path = './dataset/General/JapaneseVowels/JapaneseVowels_TEST.ts'
    # output_directory = './dataset/General/JapaneseVowels/output/'
    #
    # with open(train_file_path) as file:
    #     lines = file.readlines()
    #     i = 0
    #     for line in lines:
    #         print(line)
    #         i += 1
    #         if i == 100:
    #             break

    # # Generate sample time series
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # Compute FFT
    # fft_result = np.fft.fft(time_series)
    # freq = np.fft.fftfreq(time_series.size)
    #
    # # Consider only the positive frequencies
    # positive_freq_indices = np.where(freq > 0)
    # positive_freq = freq[positive_freq_indices]
    # positive_fft_result = fft_result[positive_freq_indices]
    #
    # # Extract the dominant period
    # dominant_frequency = positive_freq[np.argmax(np.abs(positive_fft_result))]
    # dominant_period = 1 / dominant_frequency
    #
    # # Extract the trend component
    # trend_component = np.fft.ifft(np.where(np.abs(freq) < 0.1, fft_result, 0)).real
    #
    # # Print the dominant period
    # print(f"Dominant period: {dominant_period:.2f} time units")
    #
    # # Visualize the results
    # plt.figure(figsize=(14, 7))
    #
    # # Original time series
    # plt.subplot(2, 2, 1)
    # plt.plot(time, time_series, label='Original time series')
    # plt.legend()
    #
    # # Frequency spectrum
    # plt.subplot(2, 2, 2)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # # Extracted trend
    # plt.subplot(2, 2, 3)
    # plt.plot(time, trend_component, label='Extracted trend', color='orange')
    # plt.legend()
    #
    # # Original time series and extracted trend
    # plt.subplot(2, 2, 4)
    # plt.plot(time, time_series, label='Original time series')
    # plt.plot(time, trend_component, label='Extracted trend', color='orange')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # Generate sample time series with noise
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # Compute FFT
    # fft_result = np.fft.fft(time_series)
    # freq = np.fft.fftfreq(time_series.size)
    #
    # # Only consider positive frequencies
    # positive_freq_indices = np.where(freq > 0)
    # positive_freq = freq[positive_freq_indices]
    # positive_fft_result = fft_result[positive_freq_indices]
    #
    # # Identify noise characteristics
    # noise_threshold = np.percentile(np.abs(positive_fft_result), 90)
    # noise_freq_indices = np.where(np.abs(positive_fft_result) > noise_threshold)
    #
    # # Visualize the results
    # plt.figure(figsize=(14, 7))
    #
    # # Original time series
    # plt.subplot(2, 2, 1)
    # plt.plot(time, time_series, label='Original time series')
    # plt.legend()
    #
    # # Frequency spectrum
    # plt.subplot(2, 2, 2)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # # Identified noise frequencies
    # plt.subplot(2, 2, 3)
    # plt.plot(positive_freq, np.abs(positive_fft_result), label='Frequency spectrum')
    # plt.scatter(positive_freq[noise_freq_indices], np.abs(positive_fft_result)[noise_freq_indices], color='red',
    #             label='Noise frequencies')
    # plt.xlabel('Frequency')
    # plt.ylabel('Magnitude')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    # import statsmodels.api as sm
    # from statsmodels.tsa.stattools import adfuller, kpss
    # import pywt
    #
    # # 生成示例时间序列数据
    # np.random.seed(0)
    # time = np.arange(0, 400)
    # trend = 0.01 * time
    # seasonal = 10 * np.sin(2 * np.pi * time / 50)
    # noise = np.random.normal(0, 2, time.shape)
    # time_series = trend + seasonal + noise
    #
    # # 绘制时间序列图
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, time_series)
    # plt.title("Time Series")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.show()
    #
    # # ADF检验
    # adf_result = adfuller(time_series)
    # print("ADF Statistic:", adf_result[0])
    # print("p-value:", adf_result[1])
    #
    # # KPSS检验
    # kpss_result = kpss(time_series, regression='c')
    # print("KPSS Statistic:", kpss_result[0])
    # print("p-value:", kpss_result[1])
    #
    # # 自相关函数（ACF）和偏自相关函数（PACF）
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    # sm.graphics.tsa.plot_acf(time_series, lags=40, ax=ax[0])
    # sm.graphics.tsa.plot_pacf(time_series, lags=40, ax=ax[1])
    # plt.show()
    #
    # # 自适应选择FFT或小波变换
    # if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
    #     print("Signal is stationary. Using FFT.")
    #
    #     # 使用FFT计算主要周期
    #     freq_spectrum = np.fft.fft(time_series)
    #     freqs = np.fft.fftfreq(len(time_series))
    #     positive_freqs = freqs[np.where(freqs > 0)]
    #     positive_spectrum = np.abs(freq_spectrum[np.where(freqs > 0)])
    #
    #     dominant_freq_index = np.argmax(positive_spectrum)
    #     dominant_freq = positive_freqs[dominant_freq_index]
    #     dominant_period_fft = 1 / dominant_freq
    #
    #     print("Dominant Period using FFT:", dominant_period_fft)
    #
    # else:
    #     print("Signal is non-stationary. Using Wavelet Transform.")
    #
    #     # 使用Mexican Hat小波进行CWT计算主要周期
    #     widths = np.arange(1, 128)
    #     cwt_matrix, freqs = pywt.cwt(time_series, widths, 'mexh')
    #
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(cwt_matrix, extent=[0, 400, 1, 128], cmap='PRGn', aspect='auto',
    #                vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    #     plt.colorbar(label='Coefficient Value')
    #     plt.ylabel('Scale (width)')
    #     plt.xlabel('Time')
    #     plt.title('Continuous Wavelet Transform (Mexican Hat)')
    #     plt.show()
    #
    #     dominant_scale = widths[np.argmax(np.sum(np.abs(cwt_matrix), axis=1))]
    #     dominant_period_mexican_hat = dominant_scale
    #
    #     print("Dominant Period using Mexican Hat Wavelet:", dominant_period_mexican_hat)