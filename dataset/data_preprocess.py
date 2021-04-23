# generat training and testing data and link label
# data organization
# traindata/scene#####/images(fold)
#                     /label.txt (link ground truth)
# generate data list and label
import os
import numpy as np
from glob import glob
Gl3d_dir = '/home/ubuntu/py/data/data'

mo = 0.4
ct = 0.4

for_train = True
max_img = 50
if for_train:
    output_dir = './datasets/traindata_{}_MO{}orCT{}/'.format(max_img,int(mo*10),int(ct*10))
else:
    output_dir = './datasets/testdata/benchmark/gl3d_MO{}orCT{}/'.format(int(mo*10),int(ct*10))

if not os.path.exists(output_dir):
    os.makedirs(os.path.join(output_dir,'images'))
    os.makedirs(os.path.join(output_dir, 'labels'))

files = os.listdir(Gl3d_dir)
# remove '.DS_Store' file
for item in files:
    if item.startswith('.'):
        files.remove(item)

for idx,file in enumerate(files):
    #file = '000000000000000000000013'
    img_dir = os.path.join(Gl3d_dir,file,'undist_images')
    MO_dir = os.path.join(Gl3d_dir,file,'geolabel','mesh_overlap.txt')
    CT_dir = os.path.join(Gl3d_dir,file,'geolabel','common_track.txt')

    all_imgs = glob(os.path.join(img_dir,'*.jpg'))
    all_imgs.sort(key=lambda x: (x.split('/')[-1].split('.')[0]))

    # get all view mo matrix
    # max image idex number
    last_imgs_num = int(all_imgs[-1].split('/')[-1].split('.')[0])
    print(file,last_imgs_num)
    all_mo_matrix = np.zeros((last_imgs_num+1, last_imgs_num+1))
    with open(MO_dir, 'r') as fm:
        mo_data = fm.readlines()
        for line in mo_data:
            line = line.split(' ')
            if int(line[0]) > (last_imgs_num+1) or int(line[1]) > (last_imgs_num+1):
                print(line)
                continue
            all_mo_matrix[int(line[0]), int(line[1])] = all_mo_matrix[int(line[1]), int(line[0])] = float(line[2])
        #print(all_mo_matrix)

    all_ct_matrix = np.zeros((last_imgs_num + 1, last_imgs_num + 1))
    with open(CT_dir, 'r') as fc:
        ct_data = fc.readlines()
        for line in ct_data:
            line = line.split(' ')
            if int(line[0]) > (last_imgs_num+1) or int(line[1]) > (last_imgs_num+1):
                print(line)
                continue
            all_ct_matrix[int(line[0]), int(line[1])] = all_ct_matrix[int(line[1]), int(line[0])] = float(line[2])
        #print(all_ct_matrix)

    # max 100 images

    # for traindata
    if for_train:
        if len(all_imgs) < max_img:
            continue

        else:
            num = int(len(all_imgs)/max_img)

            for k in range(num):
                imgs = all_imgs[max_img*(k):max_img*(k+1)]
                file_len = len(imgs)
                mo_matrix = np.zeros((file_len, file_len))
                ct_matrix = np.zeros((file_len, file_len))

                with open(os.path.join(output_dir,'images/data_{}_ep{}.txt'.format(file,k)),'w') as fi:

                    for i in range(max_img):
                        fi.write(imgs[i].split('/')[-1])
                        fi.write('\n')

                for i in range(max_img):
                    for j in range(i,max_img):
                        index1 = int(imgs[i].split('/')[-1].split('.')[0])
                        index2 = int(imgs[j].split('/')[-1].split('.')[0])
                        #print(index1,index2)
                        mo_matrix[i, j] = mo_matrix[j, i] = all_mo_matrix[index1, index2]

                        ct_matrix[i, j] = ct_matrix[j, i] = all_ct_matrix[index1, index2]

                #linkage label mo and ct > 0.2
                mo_thre = np.where(mo_matrix < mo, 0, 1)
                ct_thre = np.where(ct_matrix < ct, 0, 1)

                # linkage label mo > 0.3 and ct > 0.2
                # linkage_label = np.where((mo_thre+ct_thre)<1.5,0,1)
                # linkage label mo > 0.3 or ct > 0.2
                linkage_label = np.where((mo_thre + ct_thre) < 1, 0, 1)

                print(sum(sum(linkage_label)))
                np.save(os.path.join(output_dir,'labels/label_{}_ep{}'.format(file,k)),linkage_label)

    else:
        if len(all_imgs) > max_img:
            continue

        else:
            imgs = all_imgs
            file_len = len(imgs)

            with open(os.path.join(output_dir, 'images/data_{}.txt'.format(file)), 'w') as fi \
                    , open(MO_dir, 'r') as fm, open(CT_dir, 'r') as fc:

                mo_data = fm.readlines()
                ct_data = fc.readlines()

                mo_matrix = np.zeros((file_len, file_len))
                ct_matrix = np.zeros((file_len, file_len))

                for line in mo_data:
                    line = line.split(' ')
                    if int(line[0]) > file_len - 1 or int(line[1]) > file_len - 1:
                        continue
                    mo_matrix[int(line[0]), int(line[1])] = mo_matrix[int(line[1]), int(line[0])] = line[2]

                for line in ct_data:
                    line = line.split(' ')
                    if int(line[0]) > file_len - 1 or int(line[1]) > file_len - 1:
                        continue
                    ct_matrix[int(line[0]), int(line[1])] = ct_matrix[int(line[1]), int(line[0])] = line[2]

                # print(mo_matrix[0],ct_matrix[0])


                for img in imgs:
                    fi.write(img.split('/')[-1])
                    fi.write('\n')

                # linkage label mo and ct > 0.2
                mo_thre = np.where(mo_matrix < mo, 0, 1)
                ct_thre = np.where(ct_matrix < ct, 0, 1)

                # linkage label mo > 0.3 and ct > 0.2
                # linkage_label = np.where((mo_thre+ct_thre)<1.5,0,1)
                # linkage label mo > 0.3 or ct > 0.2
                linkage_label = np.where((mo_thre + ct_thre) < 1, 0, 1)

                print(sum(sum(linkage_label)))
                np.save(os.path.join(output_dir, 'labels/label_{}'.format(file)), linkage_label)

