'''Data Preprocessing'''
import csv
import pandas as pd


def trans_xlsx2csv():
    data = pd.read_excel('data/Full training dataset.xlsx')
    data.to_csv('data/Full training dataset.csv')


def full_data_process(f_r_path='datasets/Full training dataset.csv'):
    with open(f_r_path, encoding='utf-8') as f_r:
        reader = csv.reader(f_r)
        next(reader)
        for line in reader:
            activity = line[4].strip()

            verb = line[5].strip()
            object_ = line[6].strip()
            process_volume = line[8].strip()
            repetitive = line[9].strip()
            # Convert float format "0.0"/"1.0" to int "0"/"1"
            repetitive = str(int(float(repetitive))) if repetitive else ''

            gsClass = line[-1].strip()

            if gsClass and gsClass != '?':
                # No_feature: activity text only (baseline)
                with open('data/Full/No_feature/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + '\n')

                # All_feature: activity + verb + object + process_volume + repetitive
                with open('data/Full/All_feature/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + ' '.join([verb, object_, process_volume, repetitive]) + '\n')

                # Single_feature: verb
                with open('data/Full/Single_feature/verb/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + verb + '\n')

                # Single_feature: object
                with open('data/Full/Single_feature/object_/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + object_ + '\n')

                # Single_feature: process_volume
                with open('data/Full/Single_feature/process_volume/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + process_volume + '\n')

                # Single_feature: repetitive
                with open('data/Full/Single_feature/repetitive/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + repetitive + '\n')


def new_data_process(f_r_path='datasets/New training dataset.csv'):
    with open(f_r_path, encoding='utf-8') as f_r:
        reader = csv.reader(f_r)
        next(reader)
        for line in reader:
            activity = line[3].strip()

            verb = line[4].strip()
            object_ = line[5].strip()
            process_volume = line[7].strip()
            repetitive = line[8].strip()
            repetitive = str(int(float(repetitive))) if repetitive else ''

            gsClass = line[-1].strip()

            if gsClass and gsClass != '?':
                with open('data/New/All_feature/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + ' '.join(
                        [verb, object_, process_volume, repetitive]) + '\n')


def old_data_process(f_r_path='datasets/Old training dataset.csv'):
    with open(f_r_path, encoding='utf-8') as f_r:
        reader = csv.reader(f_r)
        next(reader)
        for line in reader:
            activity = line[3].strip()

            verb = line[4].strip()
            object_ = line[5].strip()
            process_volume = line[7].strip()
            repetitive = line[8].strip()
            repetitive = str(int(float(repetitive))) if repetitive else ''

            gsClass = line[-1].strip()

            if gsClass and gsClass != '?':
                with open('data/Old/All_feature/%s.txt' % str(gsClass), 'a') as f_w:
                    f_w.write(activity + ' ' + ' '.join(
                        [verb, object_, process_volume, repetitive]) + '\n')


def get_feature_data(f_r_path='data/Training_dataset_new.csv'):
    with open(f_r_path, encoding='utf-8') as f_r:
        reader = csv.reader(f_r)
        next(reader)
        for line in reader:
            activity = line[3]
            old_feature_1 = str(line[4])
            old_feature_2 = str(line[5])
            old_feature_3 = str(line[6])

            new_feature_1 = str(line[7])
            new_feature_2 = str(line[8])
            new_feature_3 = str(line[9])
            new_feature_4 = str(line[10])
            new_feature_5 = str(line[11])

            gsClass = line[-5].strip()

            if gsClass and gsClass != '?':
                with open('data/data_txt_old_feature/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([old_feature_1, old_feature_2, old_feature_3])+'\n').encode('utf-8'))
                with open('data/data_txt_new_feature/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_1, new_feature_2, new_feature_3, new_feature_4, new_feature_5])+'\n').encode('utf-8'))
                with open('data/data_txt_process_number/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_1, new_feature_2, new_feature_4, new_feature_5])+'\n').encode('utf-8'))
                with open('data/data_txt_process_string/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_1, new_feature_3, new_feature_4, new_feature_5])+'\n').encode('utf-8'))
                with open('data/data_txt_digital_data_single/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_1])+'\n').encode('utf-8'))
                with open('data/data_txt_process_string_single/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_3])+'\n').encode('utf-8'))
                with open('data/data_txt_repetive_single/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_4])+'\n').encode('utf-8'))
                with open('data/data_txt_human_single/%s.txt' % str(gsClass), 'ab') as f_w:
                    f_w.write((activity + ' '.join([new_feature_5])+'\n').encode('utf-8'))


if __name__ == '__main__':
    full_data_process()
    new_data_process()
    old_data_process()