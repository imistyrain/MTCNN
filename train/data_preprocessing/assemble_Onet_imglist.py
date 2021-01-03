import os
import sys
sys.path.append(os.getcwd())
import assemble

onet_postive_file = 'anno_store/pos_48.txt'
onet_part_file = 'anno_store/part_48.txt'
onet_neg_file = 'anno_store/neg_48.txt'
onet_landmark_file = 'anno_store/landmark_48.txt'
imglist_filename = 'anno_store/imglist_anno_48.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("ONet train annotation result file path:%s" % imglist_filename)