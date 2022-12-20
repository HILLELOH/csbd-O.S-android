import os
import shutil
import random

class copy_files:

    def true_or_false(self):
        my_random_float = random.random() #choose num randomly from [0, 1]

        if my_random_float > .5:
            return True

        else:
            return False

    def is_empty(self, path_src_folder):
        dir = os.listdir(path_src_folder)
        if len(dir) == 0:
            print("Empty directory...\n")

        else:
            print("Starting to copy...\n")


    def move(self, path_src_folder, path_dest_folder, amount_files):

        # the amount files is using to choose this amoount but randomly.
        sampled_files = list()  # the list in size: amount_files of path_files to copy into path_dest_folder

        k = 0  # counter until amount_files
        check = 0

        for item in os.scandir(path_src_folder):  # scandir is an iterator which return object from the path (files)
            if item.is_dir():
                print(f'{item} is not a file...\n')
                continue

            if k<amount_files and item==None:
                return sampled_files

            if k < amount_files:  # check that thre's not more that amount_files which was insert into the list of path
                if self.true_or_false(): #true or false randomaly
                    path_file = os.path.join(path_src_folder, item.name)  # create the full path of the .txt file

                    shutil.copy(path_file, path_dest_folder)  # copy the file to another dir
                    sampled_files.append(path_file)  # append the full path of the file item
                    print(f'Copied: {path_file}')
                    print("\n")
                    k += 1

                else:
                    continue

            else:  return sampled_files


    def clean_folder(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))





if __name__ == '__main__':
    copy = copy_files()

    # #move gooddir1
    #copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\gooddir1")
    # src_gooddir1 = "D:\\computerScience\\AttackOnAndroidOS\\dataset\\benign_apps"
    # dest_gooddir1= "D:\\computerScience\\AttackOnAndroidOS\\project\\test\\gooddir1"
    # copy.move(src_gooddir1, dest_gooddir1, 20)

    # # move maldir1
    #copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\maldir1")
    # src_maldir1 = "D:\\computerScience\\AttackOnAndroidOS\\dataset\\drebin_apps"
    # dest_maldir1 = "D:\\computerScience\\AttackOnAndroidOS\\project\\test\\maldir1"
    # copy.move(src_maldir1, dest_maldir1, 20)

    # ##########################################
    #move gooddir0
    copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\gooddir0")
    src_gooddir0="D:\\computerScience\\AttackOnAndroidOS\\dataset\\0\\train"
    dest_gooddir0="D:\\computerScience\\AttackOnAndroidOS\\project\\test\\gooddir0"
    copy.move(src_gooddir0, dest_gooddir0, 30)

    # move maldir0
    copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\maldir0")
    src_maldir0="D:\\computerScience\\AttackOnAndroidOS\\dataset\\drebin_apps"
    dest_maldir0="D:\\computerScience\\AttackOnAndroidOS\\project\\test\\maldir0"
    copy.move(src_maldir0, dest_maldir0, 30)

    # move testgooddir0
    copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\testgoodir0")
    src_testgooddir0="D:\\computerScience\\AttackOnAndroidOS\\dataset\\0\\test"
    dest_testgooddir0="D:\\computerScience\\AttackOnAndroidOS\\project\\test\\testgoodir0"
    copy.move(src_testgooddir0, dest_testgooddir0, 200)

    # move testmaldir0
    copy.clean_folder("D:\\computerScience\\AttackOnAndroidOS\\project\\test\\testmaldir0")
    src_testmaldir0="D:\\computerScience\\AttackOnAndroidOS\\dataset\\drebin_apps"
    dest_testmaldir0="D:\\computerScience\\AttackOnAndroidOS\\project\\test\\testmaldir0"
    copy.move(src_testmaldir0, dest_testmaldir0, 200)





