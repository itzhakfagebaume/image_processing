import os
from PIL import Image

# 1. On utilise le dossier que vous avez créé à l'étape précédente
# Si vous avez appelé votre dossier autrement, changez le nom ici
target_dir = "the_rock"

print(f"Traitement des images dans : {target_dir}")

for filename in os.listdir(target_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG')):
        img_path = os.path.join(target_dir, filename)
        img = Image.open(img_path)

        # 2. Calcul du carré central
        width, height = img.size
        new_edge = min(width, height)
        left = (width - new_edge) / 2
        top = (height - new_edge) / 2
        right = (width + new_edge) / 2
        bottom = (height + new_edge) / 2

        # 3. Coupe et redimensionnement à 512x512
        img = img.crop((left, top, right, bottom))
        img = img.resize((512, 512), Image.Resampling.LANCZOS)

        # 4. Sauvegarde (écrase l'original pour que ce soit prêt pour l'entraînement)
        img.save(img_path)
        print(f"✅ Image traitée : {filename}")

print("\nToutes vos photos sont maintenant des carrés parfaits de 512x512 !")