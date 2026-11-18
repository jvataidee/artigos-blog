"""Treina um classificador ResNet18 para as 8 variantes do T-72 com o arcgis.learn (ArcGIS Pro).

Parâmetros da execução registrada em resultados/: batch de 252, imagens de 138 x 138 (o `chip_size=256`
é ignorado porque a imagem original é menor), validação de 30% estratificada com semente 500, reamostragem
das classes minoritárias (`oversample`), taxa de aprendizado sugerida por `lr_find` e 100 épocas com
checkpoint e sem parada antecipada.

Uso: python treinar.py CAMINHO/DA/PASTA/T72Variants
Não foi reexecutado na reorganização do artigo.
"""

from __future__ import annotations

import sys

from arcgis.learn import FeatureClassifier, prepare_data


def main() -> None:
    raiz = sys.argv[1]
    data = prepare_data(raiz, batch_size=252, dataset_type="ImageNet", chip_size=256,
                        seed=500, val_split_pct=0.3, stratify=True)
    print(data)                                    # 3.794 imagens de treino e 1.627 de validação
    data.show_batch()

    model = FeatureClassifier(data, backbone="resnet18", oversample=True)
    lr = model.lr_find()
    print("Learning Rate:", lr)                    # 0.001445439770745928
    model.fit(100, lr=lr, early_stopping=False, checkpoint=True)
    model.plot_losses()
    model.plot_confusion_matrix()
    model.plot_hard_examples(2)                    # piores imagens, com Grad-CAM
    model.save("t72_resnet18")


if __name__ == "__main__":
    main()
