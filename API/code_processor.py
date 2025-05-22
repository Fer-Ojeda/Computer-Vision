# api/code_processor.py
import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
import supervision as sv
# matplotlib.pyplot no se usa en la API para mostrar directamente, pero la función de visualización lo usa
# import matplotlib.pyplot as plt # Descomentar si ejecutas visualizar_roi... localmente
import ultralytics as ult # Usar el alias definido
import random
import colorsys
# import torch # Importar solo si una función específica lo requiere
from collections import defaultdict
import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import datetime

# --- INICIO DE CONFIGURACIÓN ---
HORIZONTAL_OVERLAP_THRESHOLD_FOR_STACKING_SUPPRESSION_DEFAULT = 0.4
VERTICAL_OFFSET_FACTOR_FOR_STACKING_DEFAULT = 0.25 # Para generar_detecciones
MIN_DETECCIONES_POR_FILA_VALIDA_DEFAULT = 1
USAR_MATCHING_AVANZADO_POR_DEFECTO_DEFAULT = True
NUM_FILAS_VISUALES_KMEANS_DEFAULT_DEFAULT = 4
CONFIDENCE_THRESHOLD_YOLO_PREDICT_DEFAULT = 0.25
CONFIDENCE_THRESHOLD_FILTER_GDO_DEFAULT = 0.3 # Para el filtro en generar_detecciones_ordenadas
IOU_THRESHOLD_NMS_GDO_DEFAULT = 0.4

CONFIGURACIONES_ANAQUELES_RELATIVAS = {
    "1": {
        "nombre_descriptivo": "Anaquel 1 (Botanas, Abarrotes)",
        "ruta_planograma_definicion": "Anaquel1.json",
        "ruta_coordenadas_referencia": "rack_detecciones (1).json", # Asumiendo que este es el de Anaquel1
        "num_filas_planograma": 4, # Número de filas físicas que se esperan en este anaquel
        "dist_threshold_match_avanzado": 75.0,
    },
    "2": {
        "nombre_descriptivo": "Anaquel 2 (Higiene, Bebés, Bebidas)",
        "ruta_planograma_definicion": "Anaquel2.json",
        "ruta_coordenadas_referencia": "rack2_detecciones.json",
        "num_filas_planograma": 4, # Físicamente son 4 filas (ej. 5,6,7,8)
        "dist_threshold_match_avanzado": 60.0,
    }
}
# --- FIN DE CONFIGURACIÓN ---

def calcular_iou(box1_xyxy, box2_xyxy):
    x1_inter = max(box1_xyxy[0], box2_xyxy[0])
    y1_inter = max(box1_xyxy[1], box2_xyxy[1])
    x2_inter = min(box1_xyxy[2], box2_xyxy[2])
    y2_inter = min(box1_xyxy[3], box2_xyxy[3])
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    area_inter = inter_width * inter_height
    area_box1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area_box2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    area_union = area_box1 + area_box2 - area_inter
    return area_inter / float(area_union + 1e-6)

def generar_detecciones_ordenadas(yolo_result_object_roi, imagen_roi_para_anotar_bgr,
                                  ruta_planograma_nombres_param, num_filas_visuales_esperadas,
                                  confidence_threshold, iou_threshold_for_suppression,
                                  horizontal_overlap_thresh_stacking, vertical_offset_factor_stacking):
    nombres_planograma_map = {}
    lista_nombres_planograma_limpios_para_match = []
    try:
        with open(ruta_planograma_nombres_param, "r", encoding='utf-8') as f:
            planograma_n = json.load(f)
        for fila_pl in planograma_n.get("filas", []):
            for prod_pl in fila_pl.get("Productos", []):
                nombre_original = prod_pl.get("nombreProducto")
                if nombre_original:
                    nombre_limpio = nombre_original.strip().lower()
                    if nombre_limpio not in nombres_planograma_map:
                         nombres_planograma_map[nombre_limpio] = nombre_original
                    lista_nombres_planograma_limpios_para_match.append(nombre_limpio)
    except Exception as e:
        print(f"Advertencia (generar_detecciones): Al cargar planograma para nombres '{ruta_planograma_nombres_param}': {e}.")

    class_names_model = getattr(yolo_result_object_roi, 'names', {})
    
    detections_after_confidence_filter = []
    if hasattr(yolo_result_object_roi, 'boxes') and yolo_result_object_roi.boxes is not None:
        for box_idx, box in enumerate(yolo_result_object_roi.boxes):
            if box.conf is None or len(box.conf) == 0: continue
            conf = box.conf[0].cpu().numpy()
            if conf < confidence_threshold: continue
            if box.xyxy is None or len(box.xyxy) == 0: continue
            coords_roi_float = box.xyxy[0].cpu().numpy()
            if box.cls is None or len(box.cls) == 0: continue
            cls_id = int(box.cls[0].cpu().numpy())
            yolo_label_raw = class_names_model.get(cls_id, f"ID_{cls_id}")
            detections_after_confidence_filter.append({
                "id_temp": box_idx, "original_cls_id": cls_id, "yolo_label_raw": yolo_label_raw,
                "coords_float_roi": coords_roi_float, "confidence": float(conf),
                "center_x_roi": (coords_roi_float[0] + coords_roi_float[2]) / 2,
                "center_y_roi": (coords_roi_float[1] + coords_roi_float[3]) / 2,
                "height_roi": coords_roi_float[3] - coords_roi_float[1],
                "width_roi": coords_roi_float[2] - coords_roi_float[0]})
    
    detections_after_confidence_filter.sort(key=lambda d: d["confidence"], reverse=True)
    selected_detections_after_nms = []
    suppressed_indices_nms = [False] * len(detections_after_confidence_filter)
    for i in range(len(detections_after_confidence_filter)):
        if suppressed_indices_nms[i]: continue
        selected_detections_after_nms.append(detections_after_confidence_filter[i])
        for j in range(i + 1, len(detections_after_confidence_filter)):
            if suppressed_indices_nms[j]: continue
            if detections_after_confidence_filter[i]["original_cls_id"] == detections_after_confidence_filter[j]["original_cls_id"]:
                iou = calcular_iou(detections_after_confidence_filter[i]["coords_float_roi"], detections_after_confidence_filter[j]["coords_float_roi"])
                if iou > iou_threshold_for_suppression: suppressed_indices_nms[j] = True
    
    final_selected_detections = []
    suppressed_indices_stacking = [False] * len(selected_detections_after_nms)
    for i in range(len(selected_detections_after_nms)):
        if suppressed_indices_stacking[i]: continue
        det_i = selected_detections_after_nms[i]
        x1_i, y1_i, _, _ = det_i["coords_float_roi"]; height_i = det_i["height_roi"]
        for j in range(len(selected_detections_after_nms)):
            if i == j or suppressed_indices_stacking[j]: continue
            det_j = selected_detections_after_nms[j]
            if det_i["original_cls_id"] != det_j["original_cls_id"]: continue
            x1_j_comp, y1_j_comp, x2_j_comp, _ = det_j["coords_float_roi"]; width_j_comp = det_j["width_roi"]
            overlap_x1, overlap_x2 = max(x1_i, x1_j_comp), min(det_i["coords_float_roi"][2], x2_j_comp)
            overlap_width = max(0, overlap_x2 - overlap_x1)
            percentage_overlap_on_j = (overlap_width / width_j_comp) if width_j_comp > 0 else 0
            if percentage_overlap_on_j > horizontal_overlap_thresh_stacking:
                is_j_above_i = (y1_j_comp < y1_i - (height_i * vertical_offset_factor_stacking))
                if is_j_above_i:
                    if det_i["confidence"] >= det_j["confidence"]: suppressed_indices_stacking[j] = True
                    elif det_j["confidence"] > det_i["confidence"]: suppressed_indices_stacking[i] = True; break
    for idx, det in enumerate(selected_detections_after_nms):
        if not suppressed_indices_stacking[idx]: final_selected_detections.append(det)

    all_detections_processed = []
    for det_data in final_selected_detections:
        yolo_label_raw = det_data["yolo_label_raw"]; yolo_label_norm = yolo_label_raw.strip().lower()
        final_label = yolo_label_raw
        if nombres_planograma_map: # Solo si se cargó el mapa de nombres
            if yolo_label_norm in nombres_planograma_map: final_label = nombres_planograma_map[yolo_label_norm]
            else:
                for pl_key in lista_nombres_planograma_limpios_para_match:
                    if yolo_label_norm in pl_key or pl_key in yolo_label_norm: final_label = nombres_planograma_map.get(pl_key, pl_key); break
        all_detections_processed.append({
            "original_cls_id": det_data["original_cls_id"], "etiqueta_json": final_label,
            "coords_int_roi": det_data["coords_float_roi"].astype(int).tolist(),
            "center_x_roi": det_data["center_x_roi"], "center_y_roi": det_data["center_y_roi"],
            "confidence": det_data["confidence"]})

    detections_by_visual_row = []
    if all_detections_processed:
        center_y_vals_roi = np.array([d["center_y_roi"] for d in all_detections_processed]).reshape(-1, 1)
        k = min(num_filas_visuales_esperadas, len(all_detections_processed)); k = max(1, k)
        if k > 0 and k <= len(all_detections_processed):
            try:
                kmeans_rows = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(center_y_vals_roi)
                visual_row_labels = kmeans_rows.labels_
                temp_visual_rows = [[] for _ in range(k)]
                for i_km, det_item in enumerate(all_detections_processed): temp_visual_rows[visual_row_labels[i_km]].append(det_item)
                for row_dets in temp_visual_rows:
                    if row_dets: detections_by_visual_row.append({"avg_y_roi": np.mean([d["center_y_roi"] for d in row_dets]), "detections": row_dets})
            except Exception as e_kmeans: print(f"KMeans visual falló (k={k}): {e_kmeans}")
        elif all_detections_processed: detections_by_visual_row.append({"avg_y_roi": np.mean(center_y_vals_roi) if center_y_vals_roi.size > 0 else 0, "detections": all_detections_processed})
    detections_by_visual_row.sort(key=lambda r: r["avg_y_roi"])

    output_json_data = {}
    name_counters = defaultdict(int)
    for visual_row_idx, row_group in enumerate(detections_by_visual_row):
        visual_row_id = visual_row_idx + 1
        sorted_detections_in_row = sorted(row_group["detections"], key=lambda d: d["center_x_roi"])
        for pos_in_row_idx, det in enumerate(sorted_detections_in_row):
            pos_in_row_id = pos_in_row_idx + 1; label_out = det["etiqueta_json"]
            sanitized_label = ''.join(c for c in label_out if c.isalnum() or c in ['-', '_']) or "Prod"
            name_counters[sanitized_label] += 1
            json_key = f"F{visual_row_id}P{pos_in_row_id}{sanitized_label}C{name_counters[sanitized_label]}"
            coords_out = det["coords_int_roi"]
            output_json_data[json_key] = {
                "etiqueta": label_out,
                "coordenadas_en_roi": {"x1": coords_out[0], "y1": coords_out[1], "x2": coords_out[2], "y2": coords_out[3]},
                "fila_visual_inferida_desde_arriba": visual_row_id,
                "confidence_modelo": det["confidence"]}
    print(f"Preproc detecciones completado (API). {len(output_json_data)} objetos.")
    return output_json_data, None

def cargar_planograma_analisis(ruta_archivo_json):
    try:
        with open(ruta_archivo_json, 'r', encoding='utf-8') as f: planogram_data_original = json.load(f)
    except Exception as e: print(f"Error cargando planograma '{ruta_archivo_json}': {e}"); return None, 0, None
    planogram_by_row_expanded = {}; N_total_filas_planograma_fisicas = 0; unique_row_numbers = []
    if "filas" in planogram_data_original:
        filas_del_json = planogram_data_original["filas"]
        unique_row_numbers = sorted(list(set(f.get("numeroFila") for f in filas_del_json if f.get("numeroFila") is not None)))
        N_total_filas_planograma_fisicas = len(unique_row_numbers)
        for fila_info in filas_del_json:
            numero_fila = fila_info.get("numeroFila")
            if numero_fila is None: continue
            productos_originales = sorted(fila_info.get("Productos", []), key=lambda x: x.get("posicionEnFila", float('inf')))
            productos_expandidos_fila = []
            current_expanded_pos = 1
            for prod_original in productos_originales:
                for _ in range(prod_original.get("cantidadDeFrentes", 1)):
                    productos_expandidos_fila.append({"nombreProducto": prod_original.get("nombreProducto"), "CB": prod_original.get("CB"), "posicionOriginalEnFila": prod_original.get("posicionEnFila"), "posicionExpandidaEnFila": current_expanded_pos})
                    current_expanded_pos += 1
            planogram_by_row_expanded[numero_fila] = productos_expandidos_fila
    return planogram_by_row_expanded, N_total_filas_planograma_fisicas, unique_row_numbers

def obtener_roi(imagen_original_cv2, x1, y1, x2, y2): # No se usa si el frontend recorta
    if imagen_original_cv2 is None: return None
    h_orig, w_orig = imagen_original_cv2.shape[:2]
    x1_abs,y1_abs,x2_abs,y2_abs = max(0,int(x1)),max(0,int(y1)),min(w_orig,int(x2)),min(h_orig,int(y2))
    if x1_abs>=x2_abs or y1_abs>=y2_abs: return None
    return imagen_original_cv2[y1_abs:y2_abs, x1_abs:x2_abs]

def cargar_detecciones_desde_memoria(data_detecciones_dict_roi, roi_width, roi_height):
    xyxy_list, class_name_list, confidence_list, class_id_list = [], [], [], []
    class_name_to_id, next_class_id = {}, 0
    for key, detection_info in data_detecciones_dict_roi.items():
        coords_roi = detection_info.get("coordenadas_en_roi", {})
        x1_r,y1_r,x2_r,y2_r = coords_roi.get("x1"),coords_roi.get("y1"),coords_roi.get("x2"),coords_roi.get("y2")
        etiqueta = detection_info.get("etiqueta")
        if None in [x1_r,y1_r,x2_r,y2_r,etiqueta]: continue
        xyxy_list.append([x1_r,y1_r,x2_r,y2_r]); class_name_list.append(etiqueta)
        confidence_list.append(detection_info.get("confidence_modelo", 0.99))
        if etiqueta not in class_name_to_id: class_name_to_id[etiqueta] = next_class_id; next_class_id+=1
        class_id_list.append(class_name_to_id[etiqueta])
    if not xyxy_list: return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(xyxy_list), confidence=np.array(confidence_list), class_id=np.array(class_id_list), data={'class_name': np.array(class_name_list)})

def clusterizar_y_asignar_filas_analisis(detections_roi_obj, k_para_kmeans, 
                                        min_validas_por_fila_clust, 
                                        claves_filas_reales_planograma):
    
    claves_filas_ordenadas_mapeo = sorted(claves_filas_reales_planograma, reverse=True)
    detected_objects_by_planograma_row_key = {clave: [] for clave in claves_filas_reales_planograma}

    # Verificar si detections_roi_obj es None o está vacío (o no tiene xyxy)
    if not detections_roi_obj or \
       not hasattr(detections_roi_obj, 'xyxy') or \
       detections_roi_obj.xyxy is None or \
       len(detections_roi_obj.xyxy) == 0:
        print("clusterizar: No hay detecciones válidas (objeto Detections vacío o sin xyxy).")
        return detected_objects_by_planograma_row_key

    center_y_values = []
    # Iteramos para construir center_y_values, comprobando el tipo de cada 'item_det'
    for det_item in detections_roi_obj:
        # Un objeto sv.Detections (incluso de una sola detección) tendrá el atributo xyxy.
        # Si es una tupla, intentaremos acceder al primer elemento.
        current_xyxy = None
        if hasattr(det_item, 'xyxy') and det_item.xyxy is not None and len(det_item.xyxy) > 0:
            # Asume que item_det es un sv.Detections de una sola detección, xyxy es (1,4)
            current_xyxy = det_item.xyxy[0] # Accede a la primera (y única) fila de coordenadas
        elif isinstance(det_item, tuple) and len(det_item) > 0 and \
             isinstance(det_item[0], (np.ndarray, list)) and len(det_item[0]) > 0:
            # Asume que item_det es una tupla y el primer elemento es el array/lista de xyxy
            # y que este array/lista también tiene una estructura [[x1,y1,x2,y2]] o similar
            if isinstance(det_item[0], np.ndarray) and det_item[0].ndim == 2: # ej. np.array([[x1,y1,x2,y2]])
                 current_xyxy = det_item[0][0]
            elif isinstance(det_item[0], list) and len(det_item[0]) == 1 and isinstance(det_item[0][0], list): # ej. [[[x1,y1,x2,y2]]]
                 current_xyxy = det_item[0][0]
            elif isinstance(det_item[0], list) and len(det_item[0]) == 4 and isinstance(det_item[0][0], (int, float)): # ej. [x1,y1,x2,y2] dentro de la tupla
                 current_xyxy = det_item[0]

        if current_xyxy is not None and len(current_xyxy) == 4:
            center_y_values.append((current_xyxy[1] + current_xyxy[3]) / 2)
        else:
            print(f"Advertencia: No se pudo extraer xyxy de un det_item: {type(det_item)}, {det_item}")
            # Podrías decidir saltar este item o asignar un valor por defecto si es crucial no fallar

    if not center_y_values:
        print("clusterizar: No se pudieron extraer centros Y de las detecciones.")
        # Podríamos asignar todo a una fila por defecto si hay detecciones pero no centros
        if detections_roi_obj and len(detections_roi_obj) > 0 and claves_filas_ordenadas_mapeo:
             detected_objects_by_planograma_row_key[min(claves_filas_ordenadas_mapeo)] = list(detections_roi_obj)
        return detected_objects_by_planograma_row_key

    actual_k = min(k_para_kmeans, len(center_y_values), len(claves_filas_ordenadas_mapeo))
    actual_k = max(1, actual_k)

    if actual_k > len(center_y_values): # No debería pasar si center_y_values no está vacío
        print(f"clusterizar: k ({actual_k}) es mayor que n_samples ({len(center_y_values)}). Ajustando k.")
        actual_k = len(center_y_values)
        if actual_k == 0: # Si después de ajustar k es 0
             if detections_roi_obj and len(detections_roi_obj)>0 and claves_filas_ordenadas_mapeo:
                detected_objects_by_planograma_row_key[min(claves_filas_ordenadas_mapeo)] = list(detections_roi_obj)
             return detected_objects_by_planograma_row_key


    X = np.array(center_y_values).reshape(-1, 1)
    try:
        kmeans = KMeans(n_clusters=actual_k, random_state=0, n_init='auto').fit(X)
        cluster_labels = kmeans.labels_
    except Exception as e:
        print(f"Error en KMeans (clusterizar): {e}")
        if detections_roi_obj and len(detections_roi_obj)>0 and claves_filas_ordenadas_mapeo:
             detected_objects_by_planograma_row_key[min(claves_filas_ordenadas_mapeo)] = list(detections_roi_obj)
        return detected_objects_by_planograma_row_key

    clustered_detections_temp = {k_label: [] for k_label in range(actual_k)}
    # detections_roi_obj es el objeto sv.Detections original
    # Necesitamos iterar sobre él y usar los cluster_labels que corresponden a los center_y_values
    # Esto asume que el orden de center_y_values corresponde al orden de iteración de detections_roi_obj
    
    # Reconstruir la lista de detecciones válidas que se usaron para center_y_values
    valid_detections_for_clustering = []
    for det_item in detections_roi_obj:
        current_xyxy = None # Repetir la lógica de extracción de xyxy
        if hasattr(det_item, 'xyxy') and det_item.xyxy is not None and len(det_item.xyxy) > 0:
            current_xyxy = det_item.xyxy[0]
        elif isinstance(det_item, tuple) and len(det_item) > 0 and isinstance(det_item[0], (np.ndarray, list)) and len(det_item[0]) > 0:
            if isinstance(det_item[0], np.ndarray) and det_item[0].ndim == 2: current_xyxy = det_item[0][0]
            elif isinstance(det_item[0], list) and len(det_item[0]) == 1 and isinstance(det_item[0][0], list): current_xyxy = det_item[0][0]
            elif isinstance(det_item[0], list) and len(det_item[0]) == 4 and isinstance(det_item[0][0], (int, float)): current_xyxy = det_item[0]
        
        if current_xyxy is not None and len(current_xyxy) == 4:
            valid_detections_for_clustering.append(det_item) # Guardar el objeto original sv.Detections o la tupla

    # Ahora asignar a clusters_detections_temp usando valid_detections_for_clustering
    # y cluster_labels (que tiene la misma longitud que center_y_values y valid_detections_for_clustering)
    if len(valid_detections_for_clustering) == len(cluster_labels):
        for i, det_obj_original_type in enumerate(valid_detections_for_clustering):
            clustered_detections_temp[cluster_labels[i]].append(det_obj_original_type)
    else:
        print("Advertencia: Discrepancia en longitud de detecciones válidas y etiquetas de cluster.")
        # Fallback simple: si hay clusters, poner todas las detecciones en el primer cluster
        if actual_k > 0 and detections_roi_obj and len(detections_roi_obj) > 0:
            clustered_detections_temp[0].extend(list(detections_roi_obj))


    cluster_info_list = []
    for kmeans_label, dets_in_cluster in clustered_detections_temp.items():
        if dets_in_cluster:
            # Recalcular avg_y con la misma lógica de extracción de xyxy
            valid_y_for_avg = []
            for d_item in dets_in_cluster:
                xyxy_d = None
                if hasattr(d_item, 'xyxy') and d_item.xyxy is not None and len(d_item.xyxy) > 0: xyxy_d = d_item.xyxy[0]
                elif isinstance(d_item, tuple) and len(d_item) > 0 and isinstance(d_item[0], (np.ndarray, list)) and len(d_item[0]) > 0:
                    if isinstance(d_item[0], np.ndarray) and d_item[0].ndim == 2: xyxy_d = d_item[0][0]
                    elif isinstance(d_item[0], list) and len(d_item[0]) == 1 and isinstance(d_item[0][0], list): xyxy_d = d_item[0][0]
                    elif isinstance(d_item[0], list) and len(d_item[0]) == 4 and isinstance(d_item[0][0], (int, float)): xyxy_d = d_item[0]

                if xyxy_d is not None and len(xyxy_d) == 4:
                    valid_y_for_avg.append((xyxy_d[1] + xyxy_d[3]) / 2)
            
            if valid_y_for_avg:
                avg_y = np.mean(valid_y_for_avg)
                cluster_info_list.append({"avg_y_roi": avg_y, "detections": dets_in_cluster, "count": len(dets_in_cluster)})
            elif dets_in_cluster: # Si hay items pero no se pudo extraer Y, usar un Y por defecto o reportar
                 cluster_info_list.append({"avg_y_roi": 0, "detections": dets_in_cluster, "count": len(dets_in_cluster)})


    cluster_info_list.sort(key=lambda c: c["avg_y_roi"])

    for i, cluster_info in enumerate(cluster_info_list):
        if i < actual_k and i < len(claves_filas_ordenadas_mapeo):
            planograma_row_key_target = claves_filas_ordenadas_mapeo[i]
            
            if cluster_info["count"] >= min_validas_por_fila_clust:
                # Ordenar por X, usando la misma lógica condicional para acceder a xyxy
                def get_center_x(d_sort):
                    xyxy_sort = None
                    if hasattr(d_sort, 'xyxy') and d_sort.xyxy is not None and len(d_sort.xyxy) > 0: xyxy_sort = d_sort.xyxy[0]
                    elif isinstance(d_sort, tuple) and len(d_sort) > 0 and isinstance(d_sort[0], (np.ndarray, list)) and len(d_sort[0]) > 0:
                        if isinstance(d_sort[0], np.ndarray) and d_sort[0].ndim == 2: xyxy_sort = d_sort[0][0]
                        elif isinstance(d_sort[0], list) and len(d_sort[0]) == 1 and isinstance(d_sort[0][0], list): xyxy_sort = d_sort[0][0]
                        elif isinstance(d_sort[0], list) and len(d_sort[0]) == 4 and isinstance(d_sort[0][0], (int, float)): xyxy_sort = d_sort[0]
                    
                    if xyxy_sort is not None and len(xyxy_sort) == 4:
                        return (xyxy_sort[0] + xyxy_sort[2]) / 2
                    return float('inf') # Para que los no válidos vayan al final

                detected_objects_by_planograma_row_key[planograma_row_key_target] = sorted(
                    cluster_info["detections"], key=get_center_x
                )
    print(f"Clustering y asignación a filas del planograma completado. {len(detected_objects_by_planograma_row_key)} filas con posibles detecciones.")
    return detected_objects_by_planograma_row_key

def get_xyxy_from_detection_item(det_item):
    """
    Función auxiliar para obtener xyxy de forma robusta.
    Devuelve el array xyxy (ej. [x1,y1,x2,y2]) o None.
    """
    if hasattr(det_item, 'xyxy') and det_item.xyxy is not None and len(det_item.xyxy) > 0:
        # Asume sv.Detections de una sola detección, xyxy es (1,4)
        return det_item.xyxy[0]
    elif isinstance(det_item, tuple) and len(det_item) > 0:
        # Asume que el primer elemento de la tupla es el array/lista de xyxy
        if isinstance(det_item[0], np.ndarray) and det_item[0].ndim == 2 and det_item[0].shape[0] == 1 and det_item[0].shape[1] == 4: # ej. np.array([[x1,y1,x2,y2]])
             return det_item[0][0]
        elif isinstance(det_item[0], list) and len(det_item[0]) == 1 and isinstance(det_item[0][0], list) and len(det_item[0][0]) == 4: # ej. [[[x1,y1,x2,y2]]]
             return det_item[0][0]
        elif isinstance(det_item[0], list) and len(det_item[0]) == 4 and all(isinstance(c, (int, float)) for c in det_item[0]): # ej. [x1,y1,x2,y2] directamente en la tupla
             return det_item[0]
    # Si es un array NumPy directamente y tiene la forma correcta (1,4) o (4,)
    elif isinstance(det_item, np.ndarray):
        if det_item.ndim == 2 and det_item.shape[0] == 1 and det_item.shape[1] == 4: # (1,4)
            return det_item[0]
        elif det_item.ndim == 1 and det_item.shape[0] == 4: # (4,)
            return det_item
    print(f"Advertencia get_xyxy: No se pudo extraer xyxy de: {type(det_item)}, {det_item}")
    return None

def get_confidence_from_detection_item(det_item):
    """
    Función auxiliar para obtener la confianza de forma robusta.
    """
    if hasattr(det_item, 'confidence') and det_item.confidence is not None and len(det_item.confidence) > 0:
        return det_item.confidence[0]
    elif isinstance(det_item, tuple) and len(det_item) > 2 and isinstance(det_item[2], (float, np.float32)): # Asumiendo conf es el 3er elemento
        return det_item[2]
    return None # O un valor default como 0.0

def get_class_name_from_detection_item(det_item):
    """
    Función auxiliar para obtener el class_name de forma robusta.
    """
    if hasattr(det_item, 'data') and isinstance(det_item.data, dict) and \
       'class_name' in det_item.data and det_item.data['class_name'] is not None:
        # Para sv.Detections, data['class_name'] puede ser un array numpy si hay múltiples detecciones
        # o un string si es una detección individual construida de cierta manera.
        # Si es un array, tomamos el primer elemento.
        cn = det_item.data['class_name']
        return cn[0] if isinstance(cn, (np.ndarray, list)) and len(cn) > 0 else cn
    elif isinstance(det_item, tuple) and len(det_item) > 5 and \
         isinstance(det_item[5], dict) and 'class_name' in det_item[5]: # Asumiendo data es el 6to elemento y es un dict
        return det_item[5]['class_name']
    return "Desconocido"


def suprimir_detecciones_ocultas_con_prioridad_planograma(
    detecciones_asignadas_por_fila, planograma_expandido, 
    horizontal_overlap_threshold, vertical_y1_diff_threshold_factor
):
    print(f"\n--- Iniciando supresión con prioridad de planograma (H_overlap={horizontal_overlap_threshold}, V_factor={vertical_y1_diff_threshold_factor}) ---")
    detecciones_filtradas_por_fila = {fp_num: [] for fp_num in detecciones_asignadas_por_fila.keys()}

    for fila_num, dets_en_fila in detecciones_asignadas_por_fila.items():
        if not dets_en_fila: 
            continue

        productos_esperados_en_fila = planograma_expandido.get(fila_num, [])
        indices_a_mantener = list(range(len(dets_en_fila)))

        for i in range(len(dets_en_fila)):
            if i not in indices_a_mantener: 
                continue

            det_i_original_type = dets_en_fila[i]
            xyxy_i = get_xyxy_from_detection_item(det_i_original_type)

            if xyxy_i is None: # No se pudieron obtener coordenadas para det_i
                print(f"Advertencia suprimir: No se pudo obtener xyxy para det_i en fila {fila_num}, índice {i}")
                continue 
            
            x1_i, y1_i, x2_i, y2_i = xyxy_i
            height_i = y2_i - y1_i
            
            label_i = get_class_name_from_detection_item(det_i_original_type)
            conf_i = get_confidence_from_detection_item(det_i_original_type)

            prod_esp_i = productos_esperados_en_fila[i]["nombreProducto"] if i < len(productos_esperados_en_fila) else None
            es_i_correcta = (prod_esp_i is not None and label_i.strip().lower() == prod_esp_i.strip().lower())

            for j in range(len(dets_en_fila)):
                if i == j or j not in indices_a_mantener: 
                    continue

                det_j_original_type = dets_en_fila[j]
                xyxy_j = get_xyxy_from_detection_item(det_j_original_type)

                if xyxy_j is None: # No se pudieron obtener coordenadas para det_j
                    print(f"Advertencia suprimir: No se pudo obtener xyxy para det_j en fila {fila_num}, índice {j}")
                    continue

                x1_j, y1_j_coord, x2_j, y2_j_coord = xyxy_j
                width_j = x2_j - x1_j
                conf_j = get_confidence_from_detection_item(det_j_original_type)

                overlap_x1, overlap_x2 = max(x1_i, x1_j), min(x2_i, x2_j)
                overlap_width = max(0, overlap_x2 - overlap_x1)
                perc_overlap_j = (overlap_width / width_j) if width_j > 0 else 0

                if perc_overlap_j > horizontal_overlap_threshold:
                    is_j_above_i = (y1_j_coord < y1_i - (height_i * vertical_y1_diff_threshold_factor))
                    if is_j_above_i:
                        if es_i_correcta:
                            if j in indices_a_mantener: indices_a_mantener.remove(j)
                        elif conf_i is not None and conf_j is not None: # Solo comparar confianza si ambas existen
                            if conf_i >= conf_j:
                                if j in indices_a_mantener: indices_a_mantener.remove(j)
                            else: # det_j (encima) tiene más confianza
                                if i in indices_a_mantener: indices_a_mantener.remove(i); break 
                        # Si no es correcta 'i' y no hay confianzas, no se suprime por esta regla
        
        for idx_keep in sorted(indices_a_mantener):
            detecciones_filtradas_por_fila[fila_num].append(dets_en_fila[idx_keep])
        
        if len(dets_en_fila) != len(detecciones_filtradas_por_fila[fila_num]):
            print(f"  Fila {fila_num}: {len(dets_en_fila) - len(detecciones_filtradas_por_fila[fila_num])} detecciones suprimidas.")
            
    return detecciones_filtradas_por_fila

def cargar_coordenadas_referencia_planograma(ruta_json_coords_ref_param, ancho_roi_actual, alto_roi_actual):
    try:
        with open(ruta_json_coords_ref_param, 'r', encoding='utf-8') as f: data_ref = json.load(f)
    except Exception as e: print(f"Error cargando coords ref desde '{ruta_json_coords_ref_param}': {e}"); return None
    productos_ref_procesados = []
    if data_ref and "productos" in data_ref:
        ancho_img_ref = data_ref.get("ancho_imagen_ref", 1); alto_img_ref = data_ref.get("alto_imagen_ref", 1)
        escala_x = ancho_roi_actual / ancho_img_ref; escala_y = alto_roi_actual / alto_img_ref
        for prod_ref in data_ref.get("productos", []):
            bbox = prod_ref.get("bbox_ref")
            if bbox and len(bbox) == 4:
                cx_ref = (bbox[0] + bbox[2]) / 2; cy_ref = (bbox[1] + bbox[3]) / 2
                prod_ref["centroide_esperado_roi"] = (cx_ref * escala_x, cy_ref * escala_y)
                prod_ref["bbox_esperada_roi"] = [int(bbox[0]*escala_x), int(bbox[1]*escala_y), int(bbox[2]*escala_x), int(bbox[3]*escala_y)]
                productos_ref_procesados.append(prod_ref)
    return productos_ref_procesados

def comparar_con_matching_avanzado(
    planograma_original_expandido, # Usado para obtener información detallada del producto esperado si es necesario
    detecciones_asignadas_por_fila, # Dict {fila_key: [det_item_original_type, ...]}
    coords_ref_productos_en_roi,    # Lista de dicts con info de productos esperados y sus coords ROI
    dist_threshold_match
    ):
    instrucciones_estructuradas = []

    filas_presentes = set(detecciones_asignadas_por_fila.keys())
    if coords_ref_productos_en_roi:
        for p_ref in coords_ref_productos_en_roi:
            if p_ref.get("fila_planograma_ref") is not None:
                filas_presentes.add(p_ref.get("fila_planograma_ref"))
    
    sorted_filas_presentes = sorted(list(filas_presentes))

    for fila_num_key in sorted_filas_presentes:
        dets_en_fila_actual_original_type = detecciones_asignadas_por_fila.get(fila_num_key, [])
        
        esperados_en_fila_con_coords = []
        if coords_ref_productos_en_roi:
            esperados_en_fila_con_coords = [
                p for p in coords_ref_productos_en_roi
                if p.get("fila_planograma_ref") == fila_num_key
            ]
        esperados_en_fila_con_coords.sort(key=lambda p: (p.get("posicionEnFila_ref", 0), p.get("frente_num", 0)))

        detectados_info = []
        for det_idx, det_obj_original_type in enumerate(dets_en_fila_actual_original_type):
            xyxy_det = get_xyxy_from_detection_item(det_obj_original_type)
            # No añadir a detectados_info si no tenemos coordenadas válidas
            if xyxy_det is None or len(xyxy_det) != 4: 
                print(f"Advertencia (Match Avanzado): Se omitió detección en fila {fila_num_key} por xyxy inválido: {det_obj_original_type}")
                continue 

            cx = (xyxy_det[0] + xyxy_det[2]) / 2
            cy = (xyxy_det[1] + xyxy_det[3]) / 2
            label = get_class_name_from_detection_item(det_obj_original_type)
            detectados_info.append({
                "id_original_lista": det_idx, 
                "centroide": (cx, cy), 
                "etiqueta": label, 
                "obj_original": det_obj_original_type, # Mantenemos el objeto original para acceder a xyxy luego
                "bbox_coords": xyxy_det.tolist() # Guardamos las coordenadas ya extraídas
            })

        fila_texto = f"Fila {fila_num_key}"

        if not esperados_en_fila_con_coords and not detectados_info:
            continue

        if not esperados_en_fila_con_coords: # Solo extras
            for det_info in detectados_info:
                instrucciones_estructuradas.append({
                    "id_item": f"fila{fila_num_key}extra_av{det_info['id_original_lista']}",
                    "fila_planograma": fila_texto,
                    "posicion_en_fila": f"Extra (X~{det_info['centroide'][0]:.0f})",
                    "producto_esperado": None,
                    "producto_detectado": det_info['etiqueta'],
                    "status": "EXTRA",
                    "accion": f"Retirar '{det_info['etiqueta']}'.",
                    "clase_css": "instruccion-extra",
                    "bbox_detectada_roi": det_info["bbox_coords"],
                    "bbox_esperada_roi": None
                })
            continue

        if not detectados_info: # Solo faltantes
            for esp_info in esperados_en_fila_con_coords:
                pos_texto = f"Pos.Ref {esp_info.get('posicionEnFila_ref','N/A')}, Fr.{esp_info.get('frente_num','N/A')}"
                instrucciones_estructuradas.append({
                    "id_item": f"fila{fila_num_key}falt_av{esp_info.get('posicionEnFila_ref')}_{esp_info.get('frente_num')}",
                    "fila_planograma": fila_texto, "posicion_en_fila": pos_texto,
                    "producto_esperado": esp_info['nombreProducto'],
                    "producto_detectado": None,
                    "status": "FALTANTE", "accion": f"Colocar '{esp_info['nombreProducto']}'.", "clase_css": "instruccion-faltante",
                    "bbox_detectada_roi": None,
                    "bbox_esperada_roi": esp_info.get("bbox_esperada_roi") # Ya está escalada y es una lista
                })
            continue

        cost_matrix = np.full((len(esperados_en_fila_con_coords), len(detectados_info)), np.inf)
        for i, esp in enumerate(esperados_en_fila_con_coords):
            esp_center_roi = esp.get("centroide_esperado_roi")
            if esp_center_roi is None: continue
            for j, det in enumerate(detectados_info):
                dist = np.linalg.norm(np.array(esp_center_roi) - np.array(det["centroide"]))
                cost = dist
                if esp["nombreProducto"].strip().lower() != det["etiqueta"].strip().lower(): cost += 1000
                cost_matrix[i, j] = cost
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        asignaciones_esperado_a_detectado = [-1] * len(esperados_en_fila_con_coords)
        detectados_asignados_mask = [False] * len(detectados_info)

        for r_idx, c_idx in zip(row_ind, col_ind):
            esp_info = esperados_en_fila_con_coords[r_idx]
            det_info = detectados_info[c_idx]
            esp_center_roi = esp_info.get("centroide_esperado_roi")
            if esp_center_roi is None: continue

            dist_real = np.linalg.norm(np.array(esp_center_roi) - np.array(det_info["centroide"]))
            
            pos_texto = f"Pos.Ref {esp_info.get('posicionEnFila_ref','N/A')}, Fr.{esp_info.get('frente_num','N/A')}"
            item_id_base = f"fila{fila_num_key}match_av{esp_info.get('posicionEnFila_ref',r_idx)}_{esp_info.get('frente_num',0)}"
            
            current_instruction = {
                "fila_planograma": fila_texto, "posicion_en_fila": pos_texto,
                "producto_esperado": esp_info['nombreProducto'],
                "producto_detectado": det_info['etiqueta'],
                "bbox_detectada_roi": det_info["bbox_coords"],
                "bbox_esperada_roi": esp_info.get("bbox_esperada_roi")
            }

            if dist_real < dist_threshold_match:
                asignaciones_esperado_a_detectado[r_idx] = c_idx
                detectados_asignados_mask[c_idx] = True
                if esp_info["nombreProducto"].strip().lower() == det_info["etiqueta"].strip().lower():
                    current_instruction.update({
                        "id_item": item_id_base + "_ok", "status": "OK",
                        "accion": "Todo en orden.", "clase_css": "instruccion-ok"})
                else:
                    current_instruction.update({
                        "id_item": item_id_base + "_err", "status": "ERROR",
                        "accion": f"Corregir. Reemplazar '{det_info['etiqueta']}' por '{esp_info['nombreProducto']}'.",
                        "clase_css": "instruccion-error"})
                instrucciones_estructuradas.append(current_instruction)
        
        for i, esp_info in enumerate(esperados_en_fila_con_coords):
            if asignaciones_esperado_a_detectado[i] == -1:
                pos_texto = f"Pos.Ref {esp_info.get('posicionEnFila_ref','N/A')}, Fr.{esp_info.get('frente_num','N/A')}"
                instrucciones_estructuradas.append({
                    "id_item": f"fila{fila_num_key}falt_av{esp_info.get('posicionEnFila_ref',i)}_{esp_info.get('frente_num',0)}",
                    "fila_planograma": fila_texto, "posicion_en_fila": pos_texto,
                    "producto_esperado": esp_info['nombreProducto'], "producto_detectado": None,
                    "status": "FALTANTE", "accion": f"Colocar '{esp_info['nombreProducto']}'.", "clase_css": "instruccion-faltante",
                    "bbox_detectada_roi": None,
                    "bbox_esperada_roi": esp_info.get("bbox_esperada_roi")
                })
        
        for j, det_info in enumerate(detectados_info):
            if not detectados_asignados_mask[j]:
                instrucciones_estructuradas.append({
                    "id_item": f"fila{fila_num_key}extra_av{det_info['id_original_lista']}",
                    "fila_planograma": fila_texto,
                    "posicion_en_fila": f"Extra (X~{det_info['centroide'][0]:.0f})",
                    "producto_esperado": None, "producto_detectado": det_info['etiqueta'],
                    "status": "EXTRA", "accion": f"Retirar '{det_info['etiqueta']}'.", "clase_css": "instruccion-extra",
                    "bbox_detectada_roi": det_info["bbox_coords"],
                    "bbox_esperada_roi": None
                })
    return instrucciones_estructuradas

def comparar_y_generar_instrucciones_analisis(planograma_exp, detecciones_asig_fila_pl):
    instrucciones_estructuradas_simple = []
    
    sorted_fila_keys = sorted(list(planograma_exp.keys()))

    for num_fila_pl in sorted_fila_keys:
        prods_esperados_fila = planograma_exp.get(num_fila_pl, [])
        prods_detectados_fila_original_type = detecciones_asig_fila_pl.get(num_fila_pl, [])
        
        max_len = max(len(prods_esperados_fila), len(prods_detectados_fila_original_type))
        fila_texto = f"Fila {num_fila_pl}"

        if max_len == 0 and not prods_esperados_fila : 
            continue

        for i in range(max_len):
            exp_prod = prods_esperados_fila[i] if i < len(prods_esperados_fila) else None
            det_obj_original_type = prods_detectados_fila_original_type[i] if i < len(prods_detectados_fila_original_type) else None
            
            item_id_base = f"fila{num_fila_pl}_simple_pos{i+1}"
            pos_texto = f"Pos.Exp.{exp_prod.get('posicionExpandidaEnFila','N/A')} (Orig.{exp_prod.get('posicionOriginalEnFila','N/A')})" if exp_prod else f"Detectado Extra {i+1}"
            
            exp_nombre = exp_prod.get("nombreProducto") if exp_prod else None
            det_nombre = None
            bbox_det_simple_list = None

            if det_obj_original_type:
                det_nombre = get_class_name_from_detection_item(det_obj_original_type)
                bbox_coords = get_xyxy_from_detection_item(det_obj_original_type)
                if bbox_coords is not None:
                    bbox_det_simple_list = bbox_coords.tolist()


            status, accion, clase_css = "INFO", "No determinado", "instruccion-generica"

            if exp_prod and det_obj_original_type:
                if exp_nombre and det_nombre and exp_nombre.strip().lower() == det_nombre.strip().lower():
                    status, accion, clase_css = "OK", "Todo en orden.", "instruccion-ok"
                else:
                    status, accion, clase_css = "ERROR", f"Corregir. Esperado: '{exp_nombre}', Detectado: '{det_nombre}'.", "instruccion-error"
            elif exp_prod and not det_obj_original_type:
                status, accion, clase_css = "FALTANTE", f"Colocar '{exp_nombre}'.", "instruccion-faltante"
            elif not exp_prod and det_obj_original_type:
                status, accion, clase_css = "EXTRA", f"Retirar '{det_nombre}'.", "instruccion-extra"
            
            if status != "INFO":
                instrucciones_estructuradas_simple.append({
                    "id_item": item_id_base, 
                    "fila_planograma": fila_texto, 
                    "posicion_en_fila": pos_texto,
                    "producto_esperado": exp_nombre, 
                    "producto_detectado": det_nombre,
                    "status": status, 
                    "accion": accion, 
                    "clase_css": clase_css,
                    "bbox_detectada_roi": bbox_det_simple_list,
                    "bbox_esperada_roi": None # No la calculamos explícitamente en el modo simple
                })
    return instrucciones_estructuradas_simple

# --- FIN DE REFACTORIZACIÓN CRÍTICA ---

# --- FUNCIÓN PRINCIPAL PARA LA API (procesar_anaquel_para_api) ---
def procesar_anaquel_para_api(
    ruta_imagen_subida_abs,
    id_anaquel_seleccionado,
    path_modelo_yolo_abs,
    directorio_base_api
    ):

    print(f"--- Iniciando Procesamiento API para Anaquel ID: {id_anaquel_seleccionado} ---")
    
    config_anaquel_actual = CONFIGURACIONES_ANAQUELES_RELATIVAS.get(id_anaquel_seleccionado)
    if not config_anaquel_actual:
        return {"error": f"Configuración no encontrada para Anaquel ID: {id_anaquel_seleccionado}", "status_code": 400}

    print(f"Usando configuración para: {config_anaquel_actual['nombre_descriptivo']}")

    ruta_planograma_def_abs = os.path.join(directorio_base_api, config_anaquel_actual["ruta_planograma_definicion"])
    ruta_coords_ref_abs = os.path.join(directorio_base_api, config_anaquel_actual["ruta_coordenadas_referencia"])

    num_filas_planograma_config = config_anaquel_actual["num_filas_planograma"]
    dist_match_avanzado = config_anaquel_actual["dist_threshold_match_avanzado"]
    
    num_filas_visuales_kmeans = config_anaquel_actual.get("num_filas_visuales_kmeans", NUM_FILAS_VISUALES_KMEANS_DEFAULT_DEFAULT)
    conf_thresh_yolo_predict = config_anaquel_actual.get("confidence_threshold_yolo", CONFIDENCE_THRESHOLD_YOLO_PREDICT_DEFAULT)
    conf_thresh_filter_gdo = config_anaquel_actual.get("confidence_threshold_filter", CONFIDENCE_THRESHOLD_FILTER_GDO_DEFAULT) 
    horizontal_overlap_stacking_gdo = config_anaquel_actual.get("horizontal_overlap_stacking", HORIZONTAL_OVERLAP_THRESHOLD_FOR_STACKING_SUPPRESSION_DEFAULT)
    vertical_offset_stacking_gdo = config_anaquel_actual.get("vertical_offset_factor_stacking", VERTICAL_OFFSET_FACTOR_FOR_STACKING_DEFAULT)
    min_detecciones_fila_cluster = config_anaquel_actual.get("min_detecciones_por_fila_valida", MIN_DETECCIONES_POR_FILA_VALIDA_DEFAULT)
    usar_matching_avanzado_cfg = config_anaquel_actual.get("usar_matching_avanzado", USAR_MATCHING_AVANZADO_POR_DEFECTO_DEFAULT)
    iou_nms_gdo = config_anaquel_actual.get("iou_threshold_nms", IOU_THRESHOLD_NMS_GDO_DEFAULT)
    
    horizontal_overlap_supresion = config_anaquel_actual.get("horizontal_overlap_supresion", HORIZONTAL_OVERLAP_THRESHOLD_FOR_STACKING_SUPPRESSION_DEFAULT)
    vertical_offset_supresion = config_anaquel_actual.get("vertical_offset_supresion", VERTICAL_OFFSET_FACTOR_FOR_STACKING_DEFAULT)

    imagen_subida_cv2 = cv2.imread(ruta_imagen_subida_abs)
    if imagen_subida_cv2 is None:
        return {"error": f"No se pudo cargar la imagen desde: {ruta_imagen_subida_abs}", "status_code": 500}
    
    anaquel_roi_cv2 = imagen_subida_cv2 
    roi_h, roi_w = anaquel_roi_cv2.shape[:2]
    print(f"Dimensiones de imagen a procesar (ROI): {roi_w}x{roi_h}")

    try:
        yolo_model = ult.YOLO(path_modelo_yolo_abs) # <--- USO DE ult.YOLO
    except Exception as e:
        return {"error": f"Error al cargar el modelo YOLO: {e}", "status_code": 500}

    try:
        yolo_results_list_roi = yolo_model.predict(source=anaquel_roi_cv2, conf=conf_thresh_yolo_predict, verbose=False) 
    except Exception as e:
        return {"error": f"Error durante la predicción de YOLO: {e}", "status_code": 500}

    yolo_result_anaquel_roi = yolo_results_list_roi[0] if yolo_results_list_roi else None
    if not yolo_result_anaquel_roi or getattr(yolo_result_anaquel_roi, 'boxes', None) is None:
        print("Advertencia API: YOLO no devolvió resultados válidos (sin 'boxes').")
        class MockYOLOResult:
            def _init_(self): # Correcta indentación
                self.boxes = None
                self.names = getattr(yolo_model, 'names', {})
        yolo_result_anaquel_roi = MockYOLOResult()()

    detecciones_ordenadas_dict, _ = generar_detecciones_ordenadas(
        yolo_result_anaquel_roi,
        anaquel_roi_cv2.copy(), 
        ruta_planograma_def_abs,
        num_filas_visuales_kmeans,
        confidence_threshold=conf_thresh_filter_gdo, 
        iou_threshold_for_suppression=iou_nms_gdo, 
        horizontal_overlap_thresh_stacking=horizontal_overlap_stacking_gdo,
        vertical_offset_factor_stacking=vertical_offset_stacking_gdo
    )


    planograma_analisis_expandido, N_filas_fisicas_cargadas, claves_filas_reales = cargar_planograma_analisis(ruta_planograma_def_abs)
    if planograma_analisis_expandido is None or N_filas_fisicas_cargadas == 0 :
        return {"error": "No se pudo cargar o procesar el planograma para análisis (0 filas).", "status_code": 500}
    
    print(f"Planograma cargado con {N_filas_fisicas_cargadas} filas físicas distintas (claves: {claves_filas_reales}). Configuración indica {num_filas_planograma_config} filas para clustering.")

    coordenadas_productos_esperados_en_roi = cargar_coordenadas_referencia_planograma(
        ruta_coords_ref_abs, roi_w, roi_h
    )

    detecciones_en_roi_para_analisis = cargar_detecciones_desde_memoria(
        detecciones_ordenadas_dict, roi_w, roi_h
    )
    
    detecciones_clusterizadas_por_fila_pl = clusterizar_y_asignar_filas_analisis(
        detecciones_en_roi_para_analisis,
        num_filas_planograma_config, 
        min_detecciones_fila_cluster,
        claves_filas_reales
    )
    
    detecciones_intermedias_por_fila_pl = detecciones_clusterizadas_por_fila_pl
    if planograma_analisis_expandido:
         detecciones_intermedias_por_fila_pl = suprimir_detecciones_ocultas_con_prioridad_planograma(
            detecciones_clusterizadas_por_fila_pl,
            planograma_analisis_expandido,
            horizontal_overlap_threshold=horizontal_overlap_supresion,
            vertical_y1_diff_threshold_factor=vertical_offset_supresion
        )

    instrucciones_formateadas = []
    if usar_matching_avanzado_cfg and coordenadas_productos_esperados_en_roi:
        print("API: Usando comparación con MATCHING AVANZADO...")
        instrucciones_formateadas = comparar_con_matching_avanzado(
            planograma_analisis_expandido, 
            detecciones_intermedias_por_fila_pl,
            coordenadas_productos_esperados_en_roi,
            dist_threshold_match=dist_match_avanzado
        )
    else:
        if not coordenadas_productos_esperados_en_roi and usar_matching_avanzado_cfg:
            print("API ADVERTENCIA: Se solicitó matching avanzado pero faltan coordenadas de referencia. Usando comparación simple.")
        print("API: Usando comparación con ANÁLISIS SIMPLE SECUENCIAL...")
        instrucciones_formateadas = comparar_y_generar_instrucciones_analisis(
            planograma_analisis_expandido,
            detecciones_intermedias_por_fila_pl
        )
    
    print(f"API: Procesamiento completado. {len(instrucciones_formateadas)} instrucciones generadas.")
    return {
        "instrucciones": instrucciones_formateadas
    }