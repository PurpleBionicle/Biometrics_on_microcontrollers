import datetime
import os
import time

import cv2
import face_recognition
import sys
import numpy
from db import *


def check_liveness(eyes_count, open_eyes_count):
    """
    Функция сверяет количество найденных открытых глаз и общее число глаз (открытые + закрытые)
    Если количества максимально похожи (<5% разницы), то считаем, что проверка на liveness не пройденна
    :param eyes_count: количество найденных открытых глаз
    :param open_eyes_count: общее число глаз (открытые + закрытые)
    """
    return not (abs(eyes_count - open_eyes_count) <= 0.05 * max(abs(eyes_count), abs(open_eyes_count)))


def add_registration_log(log_index, name):
    """
    Метод логгирования регистрации, записи формируются в журнал logs_registration.txt
    :param log_index: номер события, которое протоколируется
    :param name: имя пользователя
    """
    messages = [f'[Error] попытка регистрации с существующим именем, имя = {name}\n',
                f'[Error] попытка регистрации через фотографию, имя = {name}\n',
                f'[Info] успешная регистрация, имя = {name}\n']
    "Получаем текущую дату и время"
    now = datetime.datetime.now()
    "Форматируем дату в нужный формат"
    formatted_date = now.strftime("[%Y-%m-%d %H:%M:%S]")
    with open('logs_registration.txt', 'a') as file:
        "Записываем строку в файл"
        file.write(f"{formatted_date} {messages[log_index]}")


def add_auth_log(log_index, name=''):
    """
    Метод логгирования аутентификации, записи формируются в журнал logs_authentication.txt
    :param log_index: номер события, которое протоколируется
    :param name: имя пользователя
    """
    messages = [f'[Error] аутентификация не пройдена\n',
                f'[Error] попытка аутентификации через фотографию\n',
                f'[Info] успешная аутентификация, имя = {name}\n']
    " Получаем текущую дату и время"
    now = datetime.datetime.now()
    "Форматируем дату в нужный формат"
    formatted_date = now.strftime("[%Y-%m-%d %H:%M:%S]")
    with open('logs_authentication.txt', 'a') as file:
        "Записываем строку в файл"
        file.write(f"{formatted_date} {messages[log_index]}")


def open_back_camera(id):
    """
    Метод включения фоновой камеры
    :param id: идентификатор камеры
    """
    camera = cv2.VideoCapture(id)
    while True:
        "получение кадров с камеры и настройка размеры окна вывода"
        ret, frame = camera.read()
        dim = (750, 500)
        frame = cv2.resize(frame, dim)
        "вывод на окно"
        if ret:
            cv2.imshow('back camera', frame)
        "Точка выхода по кнопке"
        if cv2.waitKey(1) == ord('q'):
            break
    "уничтожение окон"
    camera.release()
    cv2.destroyAllWindows()


def find_count_of_video():
    """
    Получение количества видео, чтобы дать номер следующему
    При регистрации записывается видео, из которого
    видео после завершения этапа регистрации удаляется, чтобы не хранить персональные данные
    """
    video_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "videos")
    content = os.listdir(video_path)
    indexes = []
    for file in content:
        "записанное видео имеет номер, вырежем расширение и получим нужный номер"
        if os.path.isfile(os.path.join(video_path, file)) and file.endswith('.mp4'):
            "отрежим расширение"
            file = file[:-4]
            indexes.append(int(file))
    return max(indexes) if len(indexes) != 0 else 0


def make_video(id_camera, name):
    """
    Метод записи видео при регистрации для записи биометрических данных
    :param id_camera: идентификатор камеры, откуда читать изображение
    :param name: имя пользователя, введенное при регистрации
    :return: номер записанного видео, результат проверки на liveness
    """
    "готовые фильтры для опознавания лиц и глаз"
    cascade_path = ('filters/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('filters/haarcascade_eye_tree_eyeglasses.xml')
    righteye_cascade = cv2.CascadeClassifier('filters/haarcascade_righteye_2splits.xml')
    "переменные для подсчета найденных глаз (открытых и закрытых или только открытых)"
    all_eyes_count = 0
    open_eyes_count = 0
    "на основе этого фильтра создадим классификатор"
    classifier = cv2.CascadeClassifier(cascade_path)
    "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
    "индекс камеры 0, так как она одна (встроенная)"
    camera = cv2.VideoCapture(id_camera)
    "переменные времени и размеров окна"
    start_time = datetime.datetime.now()
    delta_time_seconds = 0
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    "получение количества видео, чтобы дать название новому"
    index = find_count_of_video() + 1
    size = (frame_width, frame_height)
    "запись видео"
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    video_file = cv2.VideoWriter(f'videos/{index}.mp4', fourcc, 10, size)
    liveness = False
    while delta_time_seconds < 7:
        "захват видео с декодированием, frame - полученный кадр"
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        "найдем лица на камере, передав туда кадр"
        faces = classifier.detectMultiScale(
            frame,  # кадр
            scaleFactor=1.1,  # масштабирование
            minNeighbors=10,  # строгость критерия отбора (5 по документации)
            minSize=(30, 30),  # минимальный размер
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        "Обведем лица прямоугольниками по их размерам и координатам"
        "Также цвет рамка в rgb-формате и толщину линии"
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h + 20), (0, 255, 0), 2)
            cv2.putText(frame, name, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
            "найдем глаза"
            eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
            allEyes = righteye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
            "выделение открытых глаз"
            for (ex, ey, ew, eh) in eyes:
                open_eyes_count += 1
                # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            "выделение открытых и закрытых глаз"
            for (ex, ey, ew, eh) in allEyes:
                all_eyes_count += 1
                # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 40), 2)

        "Для демонстрации результата на кадре"
        cv2.imshow('registration', frame)
        "проверка на liveness"
        liveness = check_liveness(all_eyes_count, open_eyes_count)
        "сохранение в файл"
        if liveness:
            video_file.write(frame)
        "Точка выхода по кнопке"
        if cv2.waitKey(1) == ord('q'):
            break
        "изменение времени"
        current_time = datetime.datetime.now()
        delta_time_seconds = (current_time - start_time).total_seconds()

    "прекращение захвата и закрытие окон"
    camera.release()
    video_file.release()
    cv2.destroyAllWindows()
    return index, liveness


def make_photos_from_video(name, video_number):
    """
    из записанного видео делает фотографии из которых будут взяты лица для БД
    name = имя пользователя
    video_number - номер видео
    """
    cap = cv2.VideoCapture(f"videos/{video_number}.mp4")
    count = 0
    "создание папки базы данных"
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    while True:
        "чтение видео"
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 2
        if ret:
            "получение фото из видео"
            frame_id = int(round(cap.get(1)))
            k = cv2.waitKey(1)

            if frame_id % multiplier == 0:
                "сохранение фото регистрации"
                cv2.imwrite(f"dataset/{name}_{count}.jpg", frame)
                count += 1
            "выключение демонстрации окна по кнопке"
            if k == ord("q"):
                break
        else:
            break
    "уничтожение окон"
    cap.release()
    cv2.destroyAllWindows()


def do_authentification(id_camera):
    """
    Метод проведения аутентификации
    :param id_camera: идентификатор камеры
    :return: результат аудентификации с именем
    """
    images = os.listdir("dataset")
    while True:
        "готовый фильтр для опознавания лиц"
        cascade_path = ('filters/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('filters/haarcascade_eye_tree_eyeglasses.xml')
        righteye_cascade = cv2.CascadeClassifier('filters/haarcascade_righteye_2splits.xml')
        "переменные для подсчета найденных глаз (открытых и закрытых или только открытых)"
        all_eyes_count = 0
        open_eyes_count = 0
        "на основе этого фильтра создадим классификатор"
        classifier = cv2.CascadeClassifier(cascade_path)
        "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
        camera = cv2.VideoCapture(id_camera)

        "захват видео с декодированием, frame - полученный кадр"
        camera_is_ready, frame = camera.read()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            pass

        if camera_is_ready:
            "нахожеждение лица и его перекодировка в числовой вектор"
            faces_location = face_recognition.face_locations(frame)
            faces_encodings = face_recognition.face_encodings(frame, faces_location)
            "найдем лица на камере, передав туда кадр"
            faces = classifier.detectMultiScale(
                frame,  # кадр
                scaleFactor=1.1,  # масштабирование
                minNeighbors=10,  # строгость критерия отбора (5 по документации)
                minSize=(30, 30),  # минимальный размер
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                "найдем глаза"
                eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
                allEyes = righteye_cascade.detectMultiScale(roi_gray, minNeighbors=10)

                for (ex, ey, ew, eh) in eyes:
                    open_eyes_count += 1
                "выделение открытых и закрытых глаз"
                for (ex, ey, ew, eh) in allEyes:
                    all_eyes_count += 1
                "Для демонстрации результата на кадре"
                # cv2.imshow('auth', frame)
                "Далее произведем сравнение с датасетом"
                result,name = compare_face_with_db(faces_encodings[0])
                if result[0]:
                    liveness = check_liveness(open_eyes_count, all_eyes_count)
                    return_values = (True, name, liveness) if liveness else (True, 'photo', liveness)
                    return return_values
                return False, None, False


def do_authentification_with_name(id):
    """
    Метод проведение аутентификации с выводом имени (результата) на экран
    :param id: идентификатор камеры
    """
    while (True):
        "проведение аутентификации"
        result, name, liveness = do_authentification(id)
        unknown_name = 'unknown'
        "получение имени из результата"
        if name is None:
            name = unknown_name
        start_time = datetime.datetime.now()
        delta_time_seconds = 0
        "готовый фильтр для опознавания лиц"
        cascade_path = ('filters/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('filters/haarcascade_eye_tree_eyeglasses.xml')
        righteye_cascade = cv2.CascadeClassifier('filters/haarcascade_righteye_2splits.xml')
        "на основе этого фильтра создадим классификатор"
        classifier = cv2.CascadeClassifier(cascade_path)
        "то откуда читаем видео (камера или видеофайлы), в нашем случае камера"
        "индекс камеры 0, так как она одна (встроенная)"
        camera = cv2.VideoCapture(id)
        while delta_time_seconds < 5:
            "захват видео с декодированием, frame - полученный кадр"
            _, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            "найдем лица на камере, передав туда кадр"
            faces = classifier.detectMultiScale(
                frame,  # кадр
                scaleFactor=1.1,  # масштабирование
                minNeighbors=10,  # строгость критерия отбора (5 по документации)
                minSize=(30, 30),  # минимальный размер
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            "Обведем лица прямоугольниками по их размерам и координатам"
            "Также цвет рамка в rgb-формате и толщину линии"
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h + 20), (0, 255, 0), 2)
                cv2.putText(frame, name, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

                "найдем глаза"
                eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
                all_eyes = righteye_cascade.detectMultiScale(roi_gray, minNeighbors=10)
                "выделение открытых глаз"
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                "выделение открытых и закрытых глаз"
                for (ex, ey, ew, eh) in all_eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 40), 2)

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
        if not liveness:
            print(f"проверка на liveness не пройдена")
            add_auth_log(1)
        "если человек"
        if result and liveness:
            print(f"аутентификации пройдена успешно,это - {name}!")
            add_auth_log(2, name)
        elif liveness:
            print(f"неизвестный человек")
            add_auth_log(0)
        time.sleep(1)


def register_another_person(id_camera):
    """
    регистрация нового пользователя системы
    Делаем видео, из него после делаем скрины лиц и добавляем в наш dataset
    :param id_camera: идентификатор камеры
    """
    "ввод имени регистрации"
    name = input('введите имя для регистрации:')
    name = name.lower()
    "путь, куда запишем видео регистрации"
    video_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "dataset")
    content = os.listdir(video_path)
    name_is_exist = False
    "проверка на наличие имени, среди зарегистрированных"
    for file in content:
        index = file.index('_')
        if name == file[:index]:
            name_is_exist = True
            break
    if name_is_exist:
        print('пользователь существует, введите другое имя')
        add_registration_log(0, name)
    else:
        "Делаем видео и сохраняем"
        index, liveness = make_video(id_camera, name)
        if not liveness:
            "если проверка на liveness не прошла"
            print('проверка на liveness не пройдена')
            add_registration_log(1, name)
        else:
            "Делаем фото из видео"
            make_photos_from_video(name, index)
            "Удалим видео и фото, предварительно занесем в БД вектор лиц"
            add_photo_to_db(name,index)
            print('регистрация проведена успешно')
            add_registration_log(2, name)


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
