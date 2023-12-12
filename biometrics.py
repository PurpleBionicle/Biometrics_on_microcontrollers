import datetime
import os
import time
import cv2
import face_recognition
import sys


def add_registration_log(result, name):
    # Получаем текущую дату и время
    now = datetime.datetime.now()
    # Форматируем дату в нужный формат
    formatted_date = now.strftime("[%Y-%m-%d %H:%M:%S]")
    if not result:
        with open('logs_registration.txt', 'a') as file:
            # Записываем строку в файл
            file.write(f"{formatted_date} [Error] попытка регистрации, имя = {name}\n")
    else:
        with open('logs_registration.txt', 'a') as file:
            # Записываем строку в файл
            file.write(f"{formatted_date} [Info] регистрация, имя = {name}\n")


def add_auth_log(result, name=''):
    # Получаем текущую дату и время
    now = datetime.datetime.now()
    # Форматируем дату в нужный формат
    formatted_date = now.strftime("[%Y-%m-%d %H:%M:%S]")
    if not result:
        with open('logs_authentication.txt', 'a') as file:
            # Записываем строку в файл
            file.write(f"{formatted_date} [Error] аутентификация не пройдена\n")
    else:
        with open('logs_authentication.txt', 'a') as file:
            # Записываем строку в файл
            file.write(f"{formatted_date} [Info] аутентификация пройдена, имя = {name}\n")


def open_back_camera(id):
    camera = cv2.VideoCapture(id)
    while True:
        ret, frame = camera.read()
        dim = (1000, 500)
        frame = cv2.resize(frame, dim)
        if ret:
            cv2.imshow('back camera', frame)
        "Точка выхода по кнопке"
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def find_count_of_video():
    "получение количества видео, чтобы дать номер следующему"
    video_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "videos")
    content = os.listdir(video_path)
    indexes = []
    for file in content:
        if os.path.isfile(os.path.join(video_path, file)) and file.endswith('.mp4'):
            "отрежим расширение"
            file = file[:-4]
            indexes.append(int(file))

    return max(indexes) if len(indexes) != 0 else 0


def make_video(id_camera, name):
    "готовый фильтр для опознавания лиц"
    cascade_path = ('filters/haarcascade_frontalface_default.xml')
    "на основе этого фильтра создадим классификатор"
    classifier = cv2.CascadeClassifier(cascade_path)
    "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
    "индекс камеры 0, так как она одна (встроенная)"
    camera = cv2.VideoCapture(id_camera)
    # fps = camera.get(cv2.CAP_PROP_FPS)  # Получаем кадры в секунду
    # print(fps)
    # camera = cv2.VideoCapture('rtsp://admin:admin1@192.168.1.2/1')
    start_time = datetime.datetime.now()
    delta_time_seconds = 0
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    "получение количества видео"
    index = find_count_of_video() + 1
    size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
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
            cv2.rectangle(frame, (x, y), (x + width, y + height + 20), (0, 255, 0), 2)
            cv2.putText(frame, name, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        "Для демонстрации результата на кадре"
        cv2.imshow('registration', frame)

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
                count += 1

            if k == ord("q"):
                print("Q pressed, closing the app")
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def do_authentification(id_camera):
    images = os.listdir("dataset")
    while True:
        "готовый фильтр для опознавания лиц"
        cascade_path = ('filters/haarcascade_frontalface_default.xml')
        "на основе этого фильтра создадим классификатор"
        classifier = cv2.CascadeClassifier(cascade_path)
        "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
        camera = cv2.VideoCapture(id_camera)

        "захват видео с декодированием, frame - полученный кадр"
        camera_is_ready, frame = camera.read()

        if camera_is_ready:
            faces_location = face_recognition.face_locations(frame)
            faces_encodings = face_recognition.face_encodings(frame, faces_location)
            "найдем лица на камере, передав туда кадр"
            faces = classifier.detectMultiScale(
                frame,  # кадр
                scaleFactor=1.1,  # масштабирование
                minNeighbors=5,  # строгость критерия отбора (5 по документации)
                minSize=(30, 30),  # минимальный размер
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            "Для демонстрации результата на кадре"
            # cv2.imshow('auth', frame)
            "Далее произведем сравнение с датасетом"
            for (i, image) in enumerate(images):
                if cv2.waitKey(1) == ord('q'):
                    break

                face_img = face_recognition.load_image_file(f"dataset/{image}")
                try:
                    face_enc = face_recognition.face_encodings(face_img)[0]
                    "Сравнение из видео и датасета"
                    result = face_recognition.compare_faces([face_enc], faces_encodings[0])

                    if result[0]:
                        return True, image
                except:
                    pass
            return False, None


def do_authentification_with_name(id):
    result, img = do_authentification(id)
    name = 'unknown'
    if img is not None:
        name = img[:img.index('_')]
    start_time = datetime.datetime.now()
    delta_time_seconds = 0
    "готовый фильтр для опознавания лиц"
    cascade_path = ('filters/haarcascade_frontalface_default.xml')
    "на основе этого фильтра создадим классификатор"
    classifier = cv2.CascadeClassifier(cascade_path)
    "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
    "индекс камеры 0, так как она одна (встроенная)"
    camera = cv2.VideoCapture(id)
    while delta_time_seconds < 5:
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
            cv2.rectangle(frame, (x, y), (x + width, y + height + 20), (0, 255, 0), 2)
            cv2.putText(frame, name, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

        "Для демонстрации результата на кадре"
        cv2.imshow('auth', frame)

        "Точка выхода по кнопке"
        if cv2.waitKey(1) == ord('q'):
            break

        current_time = datetime.datetime.now()
        delta_time_seconds = (current_time - start_time).total_seconds()

    "прекращение захвата и закрытие окон"
    camera.release()
    cv2.destroyAllWindows()

    if result:
        print(f"это - {name}!")
        add_auth_log(True, name)
    else:
        print(f"неизвестный человек")
        add_auth_log(False)


def register_another_person(id_camera):
    "Делаем видео, из него после делаем скрины лиц и добавляем в наш dataset"
    name = input('введите имя для регистрации:')
    name = name.lower()
    video_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "dataset")
    content = os.listdir(video_path)
    name_is_exist = False
    for file in content:
        index = file.index('_')
        if name == file[:index]:
            name_is_exist = True
            break
    if name_is_exist:
        print('пользователь существует, введите другое имя')
        add_registration_log(False, name)
    else:
        "Делаем видео и сохраняем"
        index = make_video(id_camera, name)
        "Делаем фото из видео"
        make_photos_from_video(name, index)
        add_registration_log(True, name)


def main():
    """
    Точка входа в программу:
    1 режим: аутентификация - передаваемое видео сопоставляется с датасетом
    2 режим: регистрация - запись видео с последующем занесением в датасет
    """
    mode = sys.argv[1].lower()
    if mode == 'back1':
        open_back_camera('rtsp://admin:admin1@192.168.1.2/1')
    elif mode == 'back2':
        open_back_camera('rtsp://admin:admin1@192.168.1.3/1')
    elif mode == 'back3':
        open_back_camera(0)
    elif mode == 'registration':
        register_another_person(0)
    elif mode == 'auth':
        do_authentification_with_name(0)

    elif mode == 'registration_ip':
        id = 'rtsp://admin:admin1@192.168.1.2/1'
        register_another_person(id)
    elif mode == 'auth_ip':
        id = 'rtsp://admin:admin1@192.168.1.2/1'
        do_authentification_with_name(id)


if __name__ == '__main__':
    main()
