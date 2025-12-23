import numpy as np
import cv2
import os
from datetime import datetime

# ============================================================================
# 1. ПАРАМЕТРЫ КАЛИБРОВКИ КАМЕРЫ
# ============================================================================
def setup_camera_calibration():
    """Задание параметров калибровки камеры"""
    mtx = np.array([
        [2554.307419, 0.0, 681.336125],
        [0.0, 2557.119597, 597.850967],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist = np.array([-0.046462, 0.284386, 0.003340, -0.001596, 0.0], dtype=np.float32)

    print("Параметры калибровки камеры загружены и инициализированы.")
    print("Матрица внутренних параметров камеры (mtx):")
    print(mtx)
    print("\nВектор коэффициентов дисторсии (dist):")
    print(dist)

    # Сохраняем параметры в файлы
    np.save("camera_matrix.npy", mtx)
    np.save("dist_coeffs.npy", dist)

    print("\nДанные калибровки сохранены:")
    print("→ 'camera_matrix.npy' — матрица камеры")
    print("→ 'dist_coeffs.npy' — коэффициенты дисторсии")
    print("=" * 80)
    
    return mtx, dist

# ============================================================================
# 2. ДЕТЕКЦИЯ ARUCO-МАРКЕРОВ НА ИЗОБРАЖЕНИИ
# ============================================================================
def detect_aruco_markers(image_path, mtx, dist):
    """Обнаружение ArUco-маркеров на изображении и оценка их позы"""
    print(f"\nОбработка изображения: {image_path}")
    
    # Загрузка изображения
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Невозможно прочитать изображение '{image_path}'. Проверьте путь и формат файла.")
    
    # Подготовка изображения для детекции
    grayscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    
    # Инициализация детектора ArUco
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    detection_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(grayscale_image, aruco_dictionary, parameters=detection_params)
    
    # Карта физических размеров маркеров (в метрах)
    marker_physical_sizes_m = {
        3: 0.10,   # калибровочный маркер — 10 см
        10: 0.04   # целевой маркер — ~3 см (по диагонали)
    }
    
    annotated_image = raw_image.copy()
    rvecs_detected = []
    tvecs_detected = []
    distances = []
    
    if ids is None or len(ids) == 0:
        print("✗ На изображении не обнаружено ни одного ArUco-маркера.")
        return None, None, None, None, None
    else:
        print(f"✓ Обнаружено маркеров: {len(ids)} (ID: {ids.flatten().tolist()})")
        
        # Базовая геометрия маркера единичного размера
        unit_marker_corners = np.array([
            [-0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, -0.5, 0],
            [-0.5, -0.5, 0]
        ], dtype=np.float32)
        
        for idx, marker_id_array in enumerate(ids):
            marker_id = int(marker_id_array[0])
            
            if marker_id not in marker_physical_sizes_m:
                print(f"→ ID {marker_id}: размер не указан — пропускаем оценку позы.")
                rvecs_detected.append(None)
                tvecs_detected.append(None)
                distances.append(None)
                continue
            
            size_m = marker_physical_sizes_m[marker_id]
            print(f"→ ID {marker_id}: размер = {size_m * 100:.1f} см")
            
            # Масштабируем модель под реальный размер
            model_points = unit_marker_corners * size_m
            image_points = corners[idx].reshape(4, 2).astype(np.float32)
            
            # Оценка позы методом IPPE_SQUARE
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, mtx, dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if not success:
                print(f"    ✗ Не удалось решить PnP для ID {marker_id}")
                rvecs_detected.append(None)
                tvecs_detected.append(None)
                distances.append(None)
                continue
            
            rvecs_detected.append(rotation_vec)
            tvecs_detected.append(translation_vec)
            dist_to_marker = float(np.linalg.norm(translation_vec))
            distances.append(dist_to_marker)
            
            print(f"    ✓ tvec = [{translation_vec[0,0]:.3f}, {translation_vec[1,0]:.3f}, {translation_vec[2,0]:.3f}] м")
            print(f"      расстояние = {dist_to_marker:.3f} м")
            
            # Визуальное отображение
            contour = corners[idx].astype(np.int32)
            cv2.polylines(annotated_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
            
            axis_length = size_m * 0.8
            cv2.drawFrameAxes(annotated_image, mtx, dist, rotation_vec, translation_vec, axis_length)
            
            center_x, center_y = np.mean(contour[0], axis=0).astype(int)
            label = f"ID{marker_id}: {dist_to_marker:.2f} m"
            cv2.putText(
                annotated_image, label,
                (center_x - 30, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 2
            )
        
        # Вывод координат tvec
        print("\n" + "=" * 50)
        print("Координаты векторов смещения (tvec) для обнаруженных маркеров:")
        print("=" * 50)
        
        for i, marker_id_array in enumerate(ids):
            marker_id = int(marker_id_array[0])
            tvec = tvecs_detected[i]
            
            if tvec is not None:
                print(f"ID {marker_id:2d} → X = {tvec[0,0]:7.3f} м, "
                      f"Y = {tvec[1,0]:7.3f} м, Z = {tvec[2,0]:7.3f} м")
            else:
                print(f"ID {marker_id:2d} → — (поза не определена)")
        
        print("=" * 80)
        
        # Сохраняем аннотированное изображение
        cv2.imwrite("annotated_image.png", annotated_image)
        print(f"Аннотированное изображение сохранено как 'annotated_image.png'")
        
        return ids, corners, rvecs_detected, tvecs_detected, distances

# ============================================================================
# 3. ПРЕОБРАЗОВАНИЯ ДЛЯ МАРКЕРА ID=3 (КАЛИБРОВОЧНЫЙ)
# ============================================================================
def calculate_marker3_transformations(tvec_cam_opencv_to_marker3=None):
    """Вычисление преобразований для маркера ID=3"""
    print("\nРасчет преобразований для маркера ID=3 (калибровочный):")
    
    # Матрица поворота от системы координат камеры к маркеру ID=3
    R_cam_opencv_to_marker3_user_defined = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    
    tvec_cam_opencv_to_marker3 = np.array([
            [0.17172611],
            [-0.12859119],
            [1.3531446]
        ], dtype=np.float32)
    
    # Формируем однородную матрицу преобразования 4×4
    T_cam_opencv_to_marker3 = np.zeros((4, 4), dtype=np.float32)
    T_cam_opencv_to_marker3[3, 3] = 1.0
    T_cam_opencv_to_marker3[:3, :3] = R_cam_opencv_to_marker3_user_defined
    T_cam_opencv_to_marker3[:3, 3] = tvec_cam_opencv_to_marker3.ravel()
    
    print("Однородная матрица перехода от камеры к маркеру ID=3:")
    print(T_cam_opencv_to_marker3)
    
    # Преобразование от базы робота к маркеру ID=3
    t_robot_base_to_marker3_mm = np.array([-162.0, -274.0, 2.5], dtype=np.float32)
    t_robot_base_to_marker3 = t_robot_base_to_marker3_mm * 1e-3
    
    R_robot_base_to_marker3 = np.eye(3, dtype=np.float32)
    
    T_robot_base_to_marker3 = np.zeros((4, 4), dtype=np.float32)
    T_robot_base_to_marker3[3, 3] = 1.0
    T_robot_base_to_marker3[:3, :3] = R_robot_base_to_marker3
    T_robot_base_to_marker3[:3, 3] = t_robot_base_to_marker3
    
    print("\nПреобразование: База робота → маркер ID=3")
    print(T_robot_base_to_marker3)
    
    # Обратное преобразование
    T_marker3_to_robot_base = np.linalg.inv(T_robot_base_to_marker3)
    print("\nОбратное преобразование: маркер ID=3 → База робота")
    print(T_marker3_to_robot_base)
    
    # Композиция преобразований: камера → маркер → база робота
    T_cam_to_robot_base = T_cam_opencv_to_marker3 @ T_marker3_to_robot_base
    print("\nРезультирующая матрица: Камера (OpenCV) → База робота")
    print(T_cam_to_robot_base)
    
    # Обратное преобразование: от базы робота к камере
    T_robot_base_to_cam = np.linalg.inv(T_cam_to_robot_base)
    print("\nПолная матрица преобразования от Базы Робота до Камеры:")
    print(T_robot_base_to_cam)
    
    print("=" * 80)
    
    return T_cam_opencv_to_marker3, T_robot_base_to_cam

# ============================================================================
# 4. ПРЕОБРАЗОВАНИЯ ДЛЯ МАРКЕРА ID=10 (ЦЕЛЕВОЙ)
# ============================================================================
def calculate_marker10_transformations(ids, rvecs_detected, tvecs_detected):
    """Вычисление преобразований для маркера ID=10 (целевого)"""
    print("\nРасчет преобразований для маркера ID=10 (целевой):")
    
    # Поиск маркера ID=10 среди обнаруженных
    marker_10_index = None
    if ids is not None:
        for idx, current_id_array in enumerate(ids):
            if current_id_array.item() == 10:
                marker_10_index = idx
                break
    
    if marker_10_index is None:
        print("✗ Маркер ID=10 не найден среди обнаруженных.")
        return None
    
    # Извлекаем параметры позы
    rvec_cam_opencv_to_marker10_original = rvecs_detected[marker_10_index]
    tvec_cam_opencv_to_marker10 = tvecs_detected[marker_10_index]
    
    if rvec_cam_opencv_to_marker10_original is None or tvec_cam_opencv_to_marker10 is None:
        print("✗ Параметры позы для маркера ID=10 не определены.")
        return None
    
    # Пользовательская базовая матрица поворота
    R_user_base_for_marker10 = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    
    # Преобразуем вектор поворота в матрицу
    R_measured, _ = cv2.Rodrigues(rvec_cam_opencv_to_marker10_original)
    
    # Извлекаем угол рыскания (yaw) из измеренной матрицы
    yaw_rad = np.arctan2(R_measured[1, 0], R_measured[0, 0])
    
    # Строим чистое вращение вокруг оси Z на угол yaw
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R_z_yaw = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Комбинируем: сначала — пользовательская ориентация базиса, затем — наблюдаемый поворот вокруг Z
    R_cam_opencv_to_marker10 = R_user_base_for_marker10 @ R_z_yaw
    
    # Формируем однородную матрицу 4×4
    T_cam_opencv_to_marker10 = np.identity(4, dtype=np.float32)
    T_cam_opencv_to_marker10[:3, :3] = R_cam_opencv_to_marker10
    T_cam_opencv_to_marker10[:3, 3] = tvec_cam_opencv_to_marker10.squeeze()
    
    print("Преобразование: Камера (OpenCV) → маркер ID=10:")
    print(T_cam_opencv_to_marker10)
    
    # Сохраняем для дальнейшего использования
    np.save("T_cam_opencv_to_marker10.npy", T_cam_opencv_to_marker10)
    print("→ Сохранено в 'T_cam_opencv_to_marker10.npy'")
    print("=" * 80)
    
    return T_cam_opencv_to_marker10

# ============================================================================
# 5. ВЫЧИСЛЕНИЕ КООРДИНАТ МАРКЕРА ID=10 ОТНОСИТЕЛЬНО БАЗЫ РОБОТА
# ============================================================================
def calculate_marker10_in_robot_base(T_robot_base_to_cam, T_cam_to_marker10):
    """Вычисление координат маркера ID=10 относительно базы робота"""
    print("\nВычисление координат маркера ID=10 относительно базы робота:")
    
    if T_cam_to_marker10 is None:
        print("✗ Преобразование для маркера ID=10 не определено.")
        return None, None, None
    
    # Композиция преобразований: база робота → камера → маркер ID=10
    T_robot_base_to_marker10 = T_robot_base_to_cam @ T_cam_to_marker10
    
    print("\nРезультирующее преобразование: База робота → маркер ID=10")
    print(T_robot_base_to_marker10)
    
    # Извлечение координат маркера в системе базы
    translation_vector_marker10 = T_robot_base_to_marker10[0:3, 3].copy()
    X_marker10 = translation_vector_marker10[0]
    Y_marker10 = translation_vector_marker10[1]
    Z_marker10 = translation_vector_marker10[2]
    
    print(f"\nПоложение маркера ID=10 относительно базы робота:")
    print(f"   X = {X_marker10: .4f} м  ({X_marker10 * 1000:6.2f} мм)")
    print(f"   Y = {Y_marker10: .4f} м  ({Y_marker10 * 1000:6.2f} мм)")
    print(f"   Z = {Z_marker10: .4f} м  ({Z_marker10 * 1000:6.2f} мм)")
    print("=" * 80)
    
    return X_marker10, Y_marker10, Z_marker10

# ============================================================================
# 6. СОХРАНЕНИЕ КООРДИНАТ В ФАЙЛ
# ============================================================================
def save_coordinates_pose_goal(x, y, z, filename="marker_pose_goal.txt"):
    """
    Сохраняет координаты в формате для pose_goal (MoveIt/Python)
    """
    try:
        # Получаем текущую дату и время
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Формируем содержимое файла в нужном формате
        file_content = f"""{x}
{y}
"""
        # Записываем в файл
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(file_content)
        
        print(f"\n✓ Координаты успешно сохранены в файл: '{filename}'")
        print(f"Содержимое файла:")
        print(file_content)
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Ошибка при сохранении в файл: {e}")
        return False

# ============================================================================
# 7. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================
def main():
    """Основная функция выполнения всех расчетов"""
    print("=" * 80)
    print("АВТОМАТИЧЕСКАЯ ОБРАБОТКА ARUCO-МАРКЕРОВ И РАСЧЕТ КООРДИНАТ")
    print("=" * 80)
    
    # 1. Настройка параметров калибровки камеры
    mtx, dist = setup_camera_calibration()
    
    # 2. Детекция ArUco-маркеров на изображении
    image_path = "image5.png"
    if not os.path.exists(image_path):
        print(f"✗ Изображение '{image_path}' не найдено!")
        print("Пожалуйста, поместите файл 'image.png' в текущую директорию.")
        return
    
    ids, corners, rvecs_detected, tvecs_detected, distances = detect_aruco_markers(
        image_path, mtx, dist
    )
    
    # 3. Получаем tvec для маркера ID=3 (если он обнаружен)
    tvec_marker3 = None
    if ids is not None:
        for idx, marker_id_array in enumerate(ids):
            if marker_id_array.item() == 3 and tvecs_detected[idx] is not None:
                tvec_marker3 = tvecs_detected[idx]
                print(f"Найден маркер ID=3, используем его tvec для калибровки")
                break
    
    # 4. Вычисление преобразований для маркера ID=3
    T_cam_marker3, T_robot_base_to_cam = calculate_marker3_transformations(tvec_marker3)
    
    # 5. Вычисление преобразований для маркера ID=10
    T_cam_marker10 = calculate_marker10_transformations(ids, rvecs_detected, tvecs_detected)
    
    # 6. Вычисление координат маркера ID=10 относительно базы робота
    if T_cam_marker10 is not None:
        X_marker10, Y_marker10, Z_marker10 = calculate_marker10_in_robot_base(
            T_robot_base_to_cam, T_cam_marker10
        )
        
        # 7. Сохранение координат в файл
        if X_marker10 is not None:
            save_coordinates_pose_goal(X_marker10, Y_marker10, Z_marker10)
            
            print("\n" + "=" * 80)
            print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            print("=" * 80)
            print(f"Файл 'marker_pose_goal.txt' создан и содержит координаты X, Y маркера ID=10.")
        else:
            print("✗ Не удалось вычислить координаты маркера ID=10.")
    else:
        print("✗ Не удалось вычислить преобразования для маркера ID=10.")

# ============================================================================
# ЗАПУСК СКРИПТА
# ============================================================================
if __name__ == "__main__":
    main()