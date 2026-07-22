# Resumen — Predicción de resultado (contratado / no contratado)

Monitorización predictiva orientada a resultados, basada en prefijos (Teinemaa et al. 2019) con partición temporal estricta (Weytjens & De Weerdt 2021) y tratamiento del desbalance (Ceravolo et al. 2024).

## Mejor configuración global

- **Codificación:** frequency
- **Estrategia de desbalance:** none
- **Modelo:** RF
- **AUC-ROC (test):** 0.9540
- **AUC-PR:** 0.7075 · **F1:** 0.6849 · **Balanced acc.:** 0.7848 · **Precisión:** 0.8514 · **Recall:** 0.5729

## ¿SMOTE-NC o ponderación de clases?

- Mejor AUC-ROC sin tratamiento: 0.9540
- Mejor AUC-ROC con SMOTE-NC: 0.9482
- Mejor AUC-ROC con class weights: 0.9522
- **Preferible para este log:** ponderación de clases (class weights)

### AUC-ROC medio por estrategia

| strategy | mean | max |
| --- | --- | --- |
| none | 0.9481 | 0.9540 |
| classweight | 0.9439 | 0.9522 |
| smotenc | 0.9397 | 0.9482 |

## Tabla completa (ordenada por AUC-ROC)

| encoding | strategy | model | val_auc | scale_pos_weight | auc_roc | auc_pr | f1 | balanced_acc | precision | recall | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frequency | none | rf | 0.9522 | nan | 0.9540 | 0.7075 | 0.6849 | 0.7848 | 0.8514 | 0.5729 | 125837 | 422 | 1802 | 2417 |
| boolean | none | rf | 0.9508 | nan | 0.9530 | 0.6981 | 0.6624 | 0.7630 | 0.8877 | 0.5283 | 125977 | 282 | 1990 | 2229 |
| frequency | classweight | rf | 0.9490 | nan | 0.9522 | 0.6945 | 0.4500 | 0.8762 | 0.3112 | 0.8125 | 118671 | 7588 | 791 | 3428 |
| frequency | none | xgb | 0.9515 | nan | 0.9515 | 0.7085 | 0.6854 | 0.7843 | 0.8551 | 0.5719 | 125850 | 409 | 1806 | 2413 |
| frequency | none | lgbm | 0.9516 | nan | 0.9515 | 0.7174 | 0.6926 | 0.7862 | 0.8702 | 0.5753 | 125897 | 362 | 1792 | 2427 |
| boolean | classweight | rf | 0.9481 | nan | 0.9512 | 0.6893 | 0.4440 | 0.8757 | 0.3053 | 0.8132 | 118453 | 7806 | 788 | 3431 |
| frequency | classweight | lgbm | 0.9521 | 12.7540 | 0.9511 | 0.7088 | 0.4505 | 0.8810 | 0.3101 | 0.8232 | 118531 | 7728 | 746 | 3473 |
| boolean | classweight | lgbm | 0.9516 | 12.7540 | 0.9507 | 0.7061 | 0.4456 | 0.8808 | 0.3053 | 0.8244 | 118345 | 7914 | 741 | 3478 |
| boolean | none | lgbm | 0.9514 | nan | 0.9507 | 0.7120 | 0.6881 | 0.7819 | 0.8761 | 0.5665 | 125921 | 338 | 1829 | 2390 |
| boolean | smotenc | rf | 0.9523 | nan | 0.9482 | 0.6838 | 0.4409 | 0.8697 | 0.3042 | 0.8007 | 118532 | 7727 | 841 | 3378 |
| frequency | smotenc | rf | 0.9534 | nan | 0.9462 | 0.6850 | 0.4660 | 0.8661 | 0.3313 | 0.7853 | 119571 | 6688 | 906 | 3313 |
| boolean | none | xgb | 0.9506 | nan | 0.9455 | 0.7017 | 0.6808 | 0.7763 | 0.8801 | 0.5551 | 125940 | 319 | 1877 | 2342 |
| bigram | none | rf | 0.9454 | nan | 0.9447 | 0.6610 | 0.6591 | 0.7663 | 0.8564 | 0.5357 | 125880 | 379 | 1959 | 2260 |
| boolean | smotenc | xgb | 0.9521 | nan | 0.9443 | 0.6901 | 0.4451 | 0.8726 | 0.3075 | 0.8059 | 118601 | 7658 | 819 | 3400 |
| boolean | smotenc | lgbm | 0.9509 | nan | 0.9443 | 0.6995 | 0.4437 | 0.8720 | 0.3063 | 0.8049 | 118567 | 7692 | 823 | 3396 |
| frequency | smotenc | xgb | 0.9515 | nan | 0.9440 | 0.6884 | 0.4541 | 0.8686 | 0.3180 | 0.7940 | 119075 | 7184 | 869 | 3350 |
| bigram | none | lgbm | 0.9460 | nan | 0.9425 | 0.6852 | 0.6708 | 0.7722 | 0.8664 | 0.5473 | 125903 | 356 | 1910 | 2309 |
| frequency | classweight | xgb | 0.9503 | 12.7540 | 0.9425 | 0.6892 | 0.4490 | 0.8679 | 0.3130 | 0.7940 | 118905 | 7354 | 869 | 3350 |
| boolean | classweight | xgb | 0.9498 | 12.7540 | 0.9424 | 0.6897 | 0.4453 | 0.8707 | 0.3082 | 0.8016 | 118669 | 7590 | 837 | 3382 |
| bigram | classweight | lgbm | 0.9468 | 12.7540 | 0.9417 | 0.6715 | 0.4494 | 0.8638 | 0.3149 | 0.7845 | 119057 | 7202 | 909 | 3310 |
| frequency | smotenc | lgbm | 0.9529 | nan | 0.9410 | 0.7023 | 0.4765 | 0.8724 | 0.3399 | 0.7964 | 119735 | 6524 | 859 | 3360 |
| bigram | none | xgb | 0.9453 | nan | 0.9393 | 0.6753 | 0.6466 | 0.7571 | 0.8631 | 0.5169 | 125913 | 346 | 2038 | 2181 |
| bigram | classweight | rf | 0.9413 | nan | 0.9358 | 0.6492 | 0.4487 | 0.8523 | 0.3185 | 0.7589 | 119408 | 6851 | 1017 | 3202 |
| bigram | smotenc | rf | 0.9462 | nan | 0.9356 | 0.6355 | 0.4930 | 0.8335 | 0.3788 | 0.7056 | 121378 | 4881 | 1242 | 2977 |
| bigram | smotenc | lgbm | 0.9452 | nan | 0.9289 | 0.6574 | 0.5039 | 0.8378 | 0.3896 | 0.7130 | 121547 | 4712 | 1211 | 3008 |
| bigram | classweight | xgb | 0.9439 | 12.7540 | 0.9276 | 0.6241 | 0.4527 | 0.8488 | 0.3243 | 0.7497 | 119668 | 6591 | 1056 | 3163 |
| bigram | smotenc | xgb | 0.9454 | nan | 0.9247 | 0.6247 | 0.4537 | 0.8383 | 0.3300 | 0.7258 | 120043 | 6216 | 1157 | 3062 |