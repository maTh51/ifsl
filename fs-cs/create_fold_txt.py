import os
from PIL import Image
import numpy as np

# Lista de pastas onde estão as imagens
folders = [
    '/scratch/matheuspimenta/Vaihingen/tiles/masks/top_mosaic_09cm_area5/', 
    '/scratch/matheuspimenta/Vaihingen/tiles/masks/top_mosaic_09cm_area15/', 
    '/scratch/matheuspimenta/Vaihingen/tiles/masks/top_mosaic_09cm_area21/', 
    '/scratch/matheuspimenta/Vaihingen/tiles/masks/top_mosaic_09cm_area30/'
    ]
# Função para obter as classes de uma imagem e a porcentagem de cada classe
def get_classes_and_percentages(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Calcula a porcentagem de cada classe
    total_pixels = img_array.size
    unique_classes, class_counts = np.unique(img_array, return_counts=True)
    print(unique_classes)
    class_percentages = dict(zip(unique_classes, class_counts / total_pixels))

    return class_percentages

# Cria o arquivo fold0.txt
with open('data/splits/vaihingen/fold0.txt', 'w') as fold_file:
    # Itera sobre cada pasta
    for base_folder in folders:
        # Verifica se é um diretório
        if os.path.isdir(base_folder):
            # Itera sobre os arquivos na pasta
            for file_name in os.listdir(base_folder):
                # Verifica se o arquivo é uma imagem
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Obtém o caminho completo da imagem
                    image_path = os.path.join(base_folder, file_name)

                    # Obtém as classes e porcentagens presentes na imagem
                    class_percentages = get_classes_and_percentages(image_path)

                    # Escreve no arquivo fold0.txt se pelo menos 10% da classe está presente
                    for class_value, percentage in class_percentages.items():
                        # Ignora a classe 0 e escreve apenas se pelo menos 10% está presente
                        if class_value != 0 and percentage >= 0.1:
                            last_parent_folder = os.path.basename(os.path.dirname(base_folder))
                            fold_file.write(f"{last_parent_folder}/{os.path.splitext(file_name)[0]}__{class_value}\n")

