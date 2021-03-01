from pcp import radar_pcp
import numpy as np
import argparse
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
sys.path.insert(1,'/home/smarkowitz/open-radar') # needed to import openradar not in folder
import mmwave as mm


if __name__ == '__main__':
    # python create_radar_pcp_targets.py -rgb_input /mnt/data0-nfs/shared-datasets/rpca_test_data/output/csl_lobby_700/D.npy -radar_input /mnt/data0-nfs/shared-datasets/rpca_test_data/output/csl_lobby_700/radar_frames.txt -iterations 5
    parser = argparse.ArgumentParser()
    parser.add_argument("-rgb_input", help="path of input file", type=str)
    parser.add_argument("-radar_input", help="path to radar input", type=str)
    parser.add_argument("-max_shrink",default=1.1, help="max shrinkage value > 1",type=float)
    parser.add_argument("-min_shrink",default=.9, help="min shrinkage value < 1",type=float)
    parser.add_argument("-iterations", help="max number of iteration for PCP", type=int)
    parser.add_argument("-camera_mat", default="/home/smarkowitz/open-radar/f_matrix.npy", help="path to camera matrix to get azimuth angle to pixel location")
    parser.add_argument("-radar_cfg", default="/home/smarkowitz/open-radar/indoor_human_rcs.cfg", help="path to radar config file for radar processing",type=str)

    args = parser.parse_args()
    rgb_input_path = args.rgb_input
    radar_input_path = args.radar_input
    max_shrink = args.max_shrink
    min_shrink = args.min_shrink
    n_iterations = args.iterations
    camera_mat_path = args.camera_mat
    cfg_path = args.radar_cfg
    print('Shrink ' + str(max_shrink) + ' ' + str(min_shrink))

    output_dir = str(Path(rgb_input_path).parent)

    # load data
    radar_frames = pickle.load(open(radar_input_path,'rb'))
    iwr_cfg_cmd = mm.dataloader.cfg_list_from_cfg(cfg_path)  # this is the config sent to the radar
    iwr_cfg_dict = mm.dataloader.cfg_list_to_dict(iwr_cfg_cmd)  # this is the dictionary of config
    camera_mat = np.load(camera_mat_path)
    fx = camera_mat[0, 0]
    x_c = camera_mat[0, 2]
    D = np.load(rgb_input_path)
    n_frames, im_height, im_width = D.shape


    if n_frames != len(radar_frames):
        raise ValueError("Radar and RGB must have same number of frames")

    # Process radar data
    _, n_antennas, n_range_bins = radar_frames[0].shape

    n_angle_bins = 181
    num_vec, steering_vec = mm.dsp.gen_steering_vec(90, 1, n_antennas)  # steering vector is a(theta) in most literature

    radar_range_azs = []
    for ii in range(len(radar_frames)):
        test_radar_frame = radar_frames[ii]
        test_radar_cube = mm.dsp.range_processing(test_radar_frame)
        mean = test_radar_cube.mean(0)
        test_radar_cube = test_radar_cube - mean

        # (test_log_doppler_cube, test_doppler_cube) = mm.dsp.doppler_processing(test_radar_cube,
        #                                                                        num_tx_antennas=iwr_cfg_dict['numTx'], clutter_removal_enabled=False,
        #                                                                        interleaved=False, window_type_2d=mm.dsp.utils.Window.HAMMING,
        #                                                                        accumulate=False, phase_correction=True)
        #
        # test_doppler_cube = np.fft.fftshift(test_doppler_cube, axes=(2,))
        # test_log_doppler_cube = np.fft.fftshift(test_log_doppler_cube, axes=(2,))

        # plt.figure(figsize=(15, 5))
        # plt.suptitle('Frame %d' % ii)
        # plt.subplot(121)
        # plt.imshow(np.abs(test_doppler_cube[:, 0]).T, aspect='auto', origin='lower',
        #            extent=[0, 304 * range_res, -test_log_doppler_cube.shape[2] * dop_res / 2, test_log_doppler_cube.shape[2] * dop_res / 2])
        # plt.title('Mean-Subtraction Range Doppler')
        # plt.xlabel('Range (m)')
        # plt.ylabel('Doppler Vel. (m/s)')
        # plt.subplot(122)
        # plt.imshow(images[ii])
        # plt.axis('off')
        # plt.show()

        beamforming_result = np.zeros([n_range_bins, n_angle_bins], dtype=np.complex_)  # range bins x angle bins

        for jj in range(n_range_bins):
            beamforming_result[jj, :], _ = mm.dsp.aoa_capon(test_radar_cube.T[jj], steering_vec)
        beamforming_result = np.flip(beamforming_result,axis=1)

        radar_range_azs.append(beamforming_result)
        # plt.figure(figsize=(15, 10))
        # plt.tight_layout(True)
        # plt.subplot(111)
        # plt.imshow(np.abs(beamforming_result),origin='lower')
        # # cartesian coordinates (looks like real life)
        # beamforming_result = np.flip(beamforming_result,axis=1)  # dont know why we need this but we'll be using own polar transform in future
        # #
        # azimuths = np.radians(np.linspace(0, 180, 181))
        # zeniths = np.linspace(0, range_res * BINS_PROCESSED, BINS_PROCESSED)
        # r, theta = np.meshgrid(zeniths, azimuths)
        # values = beamforming_result.T
        # plt.pcolormesh(theta, r, np.log(np.abs(values)))
        # plt.grid()
        # plt.xlim([0, np.pi])
        # plt.show()

    radar_az_amps = [np.sum(np.abs(r_a), axis=0) for r_a in radar_range_azs]
    radar_az_log_amps = [np.log(a_a) for a_a in radar_az_amps]

    pixel_angles = 180 / np.pi * np.arctan((np.arange(im_width) - x_c) / fx)
    front_angles = np.arange(-90, 91)
    pixel_amplitudes = []

    for frame_idx in range(len(radar_az_amps)):
        radar_az_log_amp = radar_az_log_amps[frame_idx]

        f = interpolate.interp1d(front_angles, radar_az_log_amp)
        pixel_amps = f(pixel_angles) - np.min(radar_az_log_amps)
        pixel_amplitudes.append(pixel_amps)

        # plt.imshow(D[frame_idx], extent=[0, 1280, 0, 720])
        # plt.plot((pixel_amps) * 200, '--', c='lime', linewidth=4)
        # plt.show()


    # because high prob should have low cost for M_F

    l_bound = max_shrink
    u_bound = min_shrink

    pixel_amplitudes -= np.min(pixel_amplitudes)
    pixel_amplitudes /= np.max(pixel_amplitudes)
    pixel_cost_amplitudes = pixel_amplitudes * (u_bound - l_bound) + l_bound
    # for ii in pixel_cost_amplitudes:
    #     plt.plot(ii)
    #     plt.title('$\gamma$ for each column of image')
    #     plt.show()

    M_F = pixel_cost_amplitudes[:, :, np.newaxis] * np.ones((n_frames, im_width, im_height))
    M_F = np.swapaxes(M_F, 1, 2).reshape(n_frames, -1)
    M_F = M_F.T


    # radar PCP
    D = D.reshape(n_frames,-1).T
    L, S, (u, s, v), errors, ranks, nnzs = radar_pcp(D, M_F, maxiter=n_iterations, verbose=True)
    L = L.T.reshape(n_frames,im_height,im_width)
    S = S.T.reshape(n_frames,im_height,im_width)
    np.save(output_dir + '/S_radar_' + str(max_shrink) + '_' + str(min_shrink)+'_pcp.npy',S)
    np.save(output_dir + '/L_radar_' + str(max_shrink) + '_' + str(min_shrink)+'_pcp.npy',L)
