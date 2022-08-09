import cv2
from image_blur_score import BlurScore
import argparse
import tqdm
from centerface import CenterFace
import onnxruntime as ort
import subprocess


def get_faces_from_centerface(frame, centerface_threshold, centerface):
    h, w = frame.shape[:2]
    width = w
    r = width / float(w)
    height = int(h * r)

    x_scale = w / width
    y_scale = h / height

    dets, lms = centerface(frame, height, width, threshold=centerface_threshold)

    faces = []
    boxes = []

    for ff_idx, det in enumerate(dets):
        box, score = det[:4], det[4]
        x1 = int(box[0] * x_scale)
        y1 = int(box[1] * y_scale)
        x2 = int(box[2] * x_scale)
        y2 = int(box[3] * y_scale)

        face = frame[y1: y2, x1: x2, :]
        boxes.append([x1, y1, x2, y2])
        faces.append(face)

    return faces, boxes


def main():


    sess_options = ort.SessionOptions()
    centerface = CenterFace(ort, sess_options, landmarks=True)
    parser = argparse.ArgumentParser(description="Face_blur_detection")
    parser.add_argument('--videoFile')
    parser.add_argument('--centerface_threshold', type=float, default=0.7, help="centerface th")

    parser.add_argument('--th_Brenner', type=float, default=250000)
    parser.add_argument('--th_Laplacian', type=float, default=200)
    parser.add_argument('--th_Thenengrad', type=float, default=500)
    parser.add_argument('--th_SMD', type=float, default=100)
    parser.add_argument('--th_SMD2', type=float, default=1)
    parser.add_argument('--th_Variance', type=float, default=800)
    parser.add_argument('--th_Energy', type=float, default=50000000)
    parser.add_argument('--th_Vollath', type=float, default=14000000)
    parser.add_argument('--th_Entropy', type=float, default=4.71)
    parser.add_argument('--th_JPEG', type=float, default=11.924)
    parser.add_argument('--th_JPEG2', type=float, default=4.996)
    parser.add_argument('--th_Gaussian_Laplacian', type=float, default=100)

    args = parser.parse_args()

    blur_score = BlurScore()

    file_extension = "." + args.videoFile.split(".")[-1]
    output_video_filename = args.videoFile.replace(file_extension, '_output.mp4')
    output_audio_filename = args.videoFile.replace(file_extension, '.wav')

    video = cv2.VideoCapture(args.videoFile)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # th is the not normalized th base (blur)
    # th_Brenner_base = 5383350
    # th_Laplacian_base = 6.44
    # th_Thenengrad_base = 2650.78
    # th_SMD_base = 1982.75
    # th_SMD2_base = 15.73
    # th_Variance_base = 6592.09
    # th_Energy_base = 66658734
    # th_Vollath_base = 6521070215.58
    # th_Entropy_base = 4.71
    # th_JPEG_base = 11.924
    # th_JPEG2_base = 4.999561272269467
    # th_Gaussian_Laplacian_base = 12.60

    # if args.provisional_th_base:
    #     th_Brenner = args.th_Brenner * (th_Brenner_base * 2)
    #     th_Laplacian = args.th_Laplacian * (th_Laplacian_base * 2)
    #     th_Thenengrad = args.th_Thenengrad * (th_Thenengrad_base * 2)
    #     th_SMD = args.th_SMD * (th_SMD_base * 2)
    #     th_SMD2 = args.th_SMD2 * (th_SMD2_base * 2)
    #     th_Variance = args.th_Variance * (th_Variance_base * 2)
    #     th_Energy = args.th_Energy * (th_Energy_base * 2)
    #     th_Vollath = args.th_Vollath * (th_Vollath_base * 2)
    #     th_Entropy = args.th_Entropy * (th_Entropy_base * 2)
    #     th_JPEG = args.th_JPEG * (th_JPEG_base * 2)
    #     th_JPEG2 = args.th_JPEG2 * (th_JPEG2_base * 2)
    #     th_Gaussian_Laplacian = args.th_Gaussian_Laplacian * (th_Gaussian_Laplacian_base * 2)
    #
    # else:
    th_Brenner = args.th_Brenner
    th_Laplacian = args.th_Laplacian
    th_Thenengrad = args.th_Thenengrad
    th_SMD = args.th_SMD
    th_SMD2 = args.th_SMD2
    th_Variance = args.th_Variance
    th_Energy = args.th_Energy
    th_Vollath = args.th_Vollath
    th_Entropy = args.th_Entropy
    th_JPEG = args.th_JPEG
    th_JPEG2 = args.th_JPEG2
    th_Gaussian_Laplacian = args.th_Gaussian_Laplacian

    nested_blur_dict = {
        "Brenner": {'last_frame': '', 'th': th_Brenner},
        "Laplacian": {'last_frame': '', 'th': th_Laplacian},
        "Thenengrad": {'last_frame': '', 'th': th_Thenengrad},
        "SMD": {'last_frame': '', 'th': th_SMD},
        "SMD2": {'last_frame': '', 'th': th_SMD2},
        "Variance": {'last_frame': '', 'th': th_Variance},
        "Energy": {'last_frame': '', 'th': th_Energy},
        "Vollath": {'last_frame': '', 'th': th_Vollath},
        "Entropy": {'last_frame': '', 'th': th_Entropy},
        "JPEG": {'last_frame': '', 'th': th_JPEG},
        "JPEG2": {'last_frame': '', 'th': th_JPEG2},
        "Gaussian_Laplacian": {'last_frame': '', 'th': th_Gaussian_Laplacian}}

    for key, value in nested_blur_dict.items():
        output_video = output_video_filename.replace('_output.mp4', f'_{key}_output.mp4')
        value['video'] = cv2.VideoWriter(
            output_video
            , cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while True:
        for frame_num in tqdm.tqdm(range(int(num_frames))):
            ret, frame = video.read()

            if not ret:
                break
            faces, boxes = get_faces_from_centerface(frame,
                                                     centerface_threshold=args.centerface_threshold,
                                                     centerface=centerface)

            if not len(faces):
                for key, value in nested_blur_dict.items():
                    outVideo = value['video']
                    outVideo.write(frame.copy())
                continue


            for face, bbox in zip(faces, boxes):

                face = cv2.resize(face, (48, 48))

                #--------- _Brenner ---------
                score_Brenner = blur_score._Brenner(face)

                # --------- _Laplacian ---------
                score_Laplacian = blur_score._Laplacian(face)

                # --------- _Thenengrad ---------
                score_Thenengrad = blur_score._Thenengrad(face)

                # --------- _SMD ---------
                score_SMD = blur_score._SMD(face)

                # --------- _SMD2 ---------
                score_SMD2 = blur_score._SMD2(face)

                # --------- _Variance ---------
                score_Variance = blur_score._Variance(face)

                # --------- _Energy ---------
                score_Energy = blur_score._Energy(face)

                # --------- _Vollath ---------
                score_Vollath = blur_score._Vollath(face)

                # --------- _Entropy ---------
                score_Entropy = blur_score._Entropy(face)

                # --------- _JPEG ---------
                score_JPEG = blur_score._JPEG(face)

                # --------- _JPEG2 ---------
                score_JPEG2 = blur_score._JPEG2(face)

                # --------- _Gaussian_Laplacian ---------
                score_Gaussian_Laplacian = blur_score._Gaussian_Laplacian(face)

                nested_blur_dict['Brenner']['last_score'] = score_Brenner
                nested_blur_dict['Laplacian']['last_score'] = score_Laplacian
                nested_blur_dict['Thenengrad']['last_score'] = score_Thenengrad
                nested_blur_dict['SMD']['last_score'] = score_SMD
                nested_blur_dict['SMD2']['last_score'] = score_SMD2
                nested_blur_dict['Variance']['last_score'] = score_Variance
                nested_blur_dict['Energy']['last_score'] = score_Energy
                nested_blur_dict['Vollath']['last_score'] = score_Vollath
                nested_blur_dict['Entropy']['last_score'] = score_Entropy
                nested_blur_dict['JPEG']['last_score'] = score_JPEG
                nested_blur_dict['JPEG2']['last_score'] = score_JPEG2
                nested_blur_dict['Gaussian_Laplacian']['last_score'] = score_Gaussian_Laplacian

                for key, value in nested_blur_dict.items():

                    if value['last_frame'] == '':
                        frame_copy = frame.copy()
                    else:
                        frame_copy = value['last_frame']

                    last_blur_score = float(value['last_score'])
                    blur_th = float(value['th'])

                    if ('JPEG' in key) or ('Vollath' in key):
                        if last_blur_score <= blur_th:
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)
                    else:
                        if last_blur_score >= blur_th:
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)

                    last_blur_score = float(f"{last_blur_score:0.5f}")
                    frame_copy = cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 10)
                    cv2.putText(frame_copy, f"{last_blur_score}", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame_copy, f"BLUR METHOD: {key}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

                    value['last_frame'] = frame_copy.copy()

            for key, value in nested_blur_dict.items():
                outVideo = value['video']
                outVideo.write(value['last_frame'])
                value['last_frame'] = ''

            # if frame_num >= 20:
            #     break

        break

    for key, value in nested_blur_dict.items():
        outVideo = value['video']
        outVideo.release()

    cv2.destroyAllWindows()
    video.release()

    # subprocess.call(
    #     'ffmpeg -i %s -i %s -shortest  %s -y' % (output_video_filename,
    #                                              output_audio_filename,
    #                                              output_video_filename.replace(file_extension, "_AV.mp4")),
    #     shell=True)


if __name__ == '__main__':
    main()