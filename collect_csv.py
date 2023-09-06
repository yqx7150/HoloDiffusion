import os
from glob import glob
import pandas as pd

typeList = ['holo','phase','amp'] # phase amp holo
netList = ['NCSNPP','SRSAA'] # SRSAA

for type in typeList:
    for net in netList:
        files = glob(f'/code/HOLOGRAM/ncsnpp_holo/ncsnpp_holo_train/only_amp_phase/result_1k_m20/everything/{net}_{type}_*/{type}_all.csv')

        # P1 C8: files = glob(f'/code/ZL/NCSN_CT/NCSN_GB/test_odl_ngf64_hankel_piece/result_zl/*/P*R*cut_20.csv')

        savefile = f"./result_collect/{type}_{net}everything.csv"
        print(f"{savefile} --------------------- ")
        if os.path.exists(savefile):
            os.remove(savefile)
            print("Delete old files succeed ... Now it is a New start....")

        addHead=True
        for idx, file in enumerate(files):
            print(file)
            df = pd.read_csv(file)
            if addHead:
                df.to_csv(savefile, mode='a', index=False, header=True)
                addHead = False
            else:
                df.to_csv(savefile, mode='a', index=False, header=False)
