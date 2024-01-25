import os
import sqlite3
import time

import cv2
import face_recognition
import pickle
import pysqlcipher3
import secret


def add_photo_to_db(name, index=0):
    """
    Добавление в базу данных векторов лиц для переданного имени
    Занесение в базу осуществляется в зашифрованном виде
    :param name: имя пользователя
    :param index: номер видео,которое надо удалить
    """
    "удалим видео"
    # os.remove(f'videos/{index}')
    "подсоединимся к базе данных и возьмем управление (курсор)"
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    "подключим шифрование на нашем ключе с помощью модуля pysqlcipher3"
    connection.execute(f"ATTACH DATABASE 'database.db' AS encrypted KEY '{secret.encryption_key}';")
    "Создаем таблицу Users если ее нет"
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
    username TEXT NOT NULL,
    face_vector BLOB)
    ''')
    connection.commit()

    "все изображения из видео перенес в ДБ"
    images = os.listdir("dataset")
    for image in images:
        if name in image:
            face_img = face_recognition.load_image_file(f"dataset/{image}")
            try:
                "его декодирование"
                face_enc = face_recognition.face_encodings(face_img)[0]
                "применим сериализацию"
                face_pickle = pickle.dumps(face_enc)
                "добавим запись о пользователе"
                cursor.execute('INSERT INTO Users (username, face_vector) VALUES (?, ?)', (name, face_pickle))
                connection.commit()
                "удалим фото"
                # os.remove(f'dataset/{image}')
            except:
                pass
    connection.close()


def compare_face_with_db(face_from_camera):
    """
    Функция сравнения векторов лиц из БД и переданного
    :param face: вектор лица из видео при аутентификаци
    :return: имя пользователя
    """
    "подсоединимся к базе данных и возьмем управление (курсор)"
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    "достанем все записи из БД"
    cursor.execute('SELECT * FROM Users')
    users = cursor.fetchall()
    "сравним с нашим лицом"
    for user in users:
        try:
            face_pickle = pickle.loads(user[1])
            result = face_recognition.compare_faces([face_pickle], face_from_camera)
            if result == [True]:
                return result, user[0]
        except:
            pass
    return False, 'unknown'


for i in range(100):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    "подключим шифрование на нашем ключе с помощью модуля pysqlcipher3"
    connection.execute(f"ATTACH DATABASE 'database.db' AS encrypted KEY '{secret.encryption_key}';")
    "Создаем таблицу Users если ее нет"
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
        username TEXT NOT NULL,
        face_vector BLOB)
        ''')
    connection.commit()

    "все изображения из видео перенес в ДБ"
    cursor.execute('INSERT INTO Users (username, face_vector) VALUES (?, ?)', (i, 0))
    connection.commit()
    connection.close()
