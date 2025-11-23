 RGB_PASS='/home/usrs/ugawa/lab/work/TICCGAN/results_8/2023_9_25_gll_FLIR_resnet101/gll_FLIR_full/test_day_100/real_B'
 THERMAL_PASS='/home/usrs/ugawa/lab/work/TICCGAN/results_8/2023_9_25_gll_FLIR_resnet101/gll_FLIR_full/test_day_100/real_A'
 TARGET_PASS='/home/usrs/ugawa/lab/work/TICCGAN/results_8/2023_9_25_gll_FLIR_resnet101/gll_FLIR_full/test_day_100'
 mkdir -p ${TARGET_PASS}/L_is_thermal
 python swap_L_channel.py --dirA ${RGB_PASS} --dirB ${THERMAL_PASS} --dirC ${TARGET_PASS}/L_is_thermal
