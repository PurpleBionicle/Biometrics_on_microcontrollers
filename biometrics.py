import datetime
import os

import cv2
import face_recognition
from PIL import Image, ImageDraw


def find_count_of_video():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    content = os.listdir(video_path)
    indexes = []
    for file in content:
        if os.path.isfile(os.path.join(video_path, file)) and file.endswith('.mp4'):
            "отрежим расширение"
            file = file[:-4]
            indexes.append(int(file))

    return max(indexes) if len(indexes) != 0 else 0


def make_video():
    "готовый фильтр для опознавания лиц"
    cascade_path = ('filters/haarcascade_frontalface_default.xml')
    "на основе этого фильтра создадим классификатор"
    classifier = cv2.CascadeClassifier(cascade_path)
    "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
    "индекс камеры 0, так как она одна"
    camera = cv2.VideoCapture(0)
    start_time = datetime.datetime.now()
    delta_time_seconds = 0

    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    "получение количества видео"
    index = find_count_of_video() + 1
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter.fourcc(*'MP4V')
    video_file = cv2.VideoWriter(f'videos/{index}.mp4', fourcc, 10, size)

    while delta_time_seconds < 10:
        "захват видео с декодированием, frame - полученный кадр"
        _, frame = camera.read()
        "найдем лица на камере, передав туда кадр"
        faces = classifier.detectMultiScale(
            frame,  # кадр
            scaleFactor=1.1,  # масштабирование
            minNeighbors=3,  # строгость критерия отбора (5 по документации)
            minSize=(30, 30),  # минимальный размер
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        "Обведем лица прямоугольниками по их размерам и координатам"
        "Также цвет рамка в rgb-формате и толщину линии"
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        "Для демонстрации результата на кадре"
        cv2.imshow('Faces', frame)

        video_file.write(frame)

        "Точка выхода по кнопке"
        if cv2.waitKey(1) == ord('q'):
            break

        current_time = datetime.datetime.now()
        delta_time_seconds = (current_time - start_time).total_seconds()

    "прекращение захвата и закрытие окон"
    camera.release()
    video_file.release()
    cv2.destroyAllWindows()
    return index


def make_photos_from_video(name, video_number):
    cap = cv2.VideoCapture(f"videos/{video_number}.mp4")
    count = 0

    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 2

        if ret:
            frame_id = int(round(cap.get(1)))
            k = cv2.waitKey(1)

            if frame_id % multiplier == 0:
                cv2.imwrite(f"dataset/{name}_{count}.jpg", frame)
                print(f"Take a screenshot {count}")
                count += 1

            if k == ord("q"):
                print("Q pressed, closing the app")
                break

        else:
            print("[Error] Can't get the frame...")
            break

    cap.release()
    cv2.destroyAllWindows()


def do_authentification():
    images = os.listdir("dataset")
    while True:
        "готовый фильтр для опознавания лиц"
        cascade_path = ('filters/haarcascade_frontalface_default.xml')
        "на основе этого фильтра создадим классификатор"
        classifier = cv2.CascadeClassifier(cascade_path)
        "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
        "индекс камеры 0, так как она одна"
        camera = cv2.VideoCapture(0)

        frame_width = int(camera.get(3))
        frame_height = int(camera.get(4))
        size = (frame_width, frame_height)
        "захват видео с декодированием, frame - полученный кадр"
        camera_is_ready, frame = camera.read()
        if camera_is_ready:
            faces_location = face_recognition.face_locations(frame)
            faces_encodings = face_recognition.face_encodings(frame, faces_location)
            "найдем лица на камере, передав туда кадр"
            faces = classifier.detectMultiScale(
                frame,  # кадр
                scaleFactor=1.1,  # масштабирование
                minNeighbors=3,  # строгость критерия отбора (5 по документации)
                minSize=(30, 30),  # минимальный размер
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            "Для демонстрации результата на кадре"
            cv2.imshow('Faces', frame)

            "Далее произведем сравнение с датасетом"
            for (i, image) in enumerate(images):

                if cv2.waitKey(1) == ord('q'):
                    break

                print(f"[+] processing img {i + 1}/{len(images)}")

                face_img = face_recognition.load_image_file(f"dataset/{image}")
                try:
                    face_enc = face_recognition.face_encodings(face_img)[0]
                    "Сравнение из видео и датасета"
                    result = face_recognition.compare_faces([face_enc], faces_encodings[0])
                    # print(result)

                    if result[0]:
                        return True, image
                except:
                    pass
            return False, None


def register_another_person():
    "Делаем видео, из него после делаем скрины лиц и добавляем в наш dataset"
    name = input('your name to login:')
    index = make_video()
    make_photos_from_video(name, index)


def main():
    work_type = int(input('type of work:'))
    if work_type == 1:
        result, img = do_authentification()
        if result:
            print(f" same people with {img}!")
        else:
            print(f"unknown people")

    elif work_type == 2:
        register_another_person()


if __name__ == '__main__':
    main()
