import glob
import numpy as np

class Load_data:
    def __init__(self, _data, _labels):
        self._dir = (_data, _labels)

    def get_labeled_data(self) -> dict:
        """
        Function to get labeled data in a _ format end-to-end.
        get dict of ID of data, Text, and Label (binary for SLC).

        Args:
            Class.

        Returns:
            Dict of ID, Text, Label.
        """
        # get 2 dirs - data and label
        self.list_of_articles = glob.glob(f'{self._dir[0]}\*.txt') # text
        self.list_of_labels = glob.glob(f'{self._dir[1]}\*.labels') # labels

        self.article_ids = []
        self.article_text = []
        self.article_label = []

        self.dataset_full = {}

        # Step 1: Get all ids
        # Step 2: Match id with .txt and make it a vector
        # Step 3: Match id with .label and make the 3rd col a vector
        # Step 4: Export this as final data - (All combined or per id?)

        # Get all labels
        for file_name in self.list_of_articles: # loop over file w unique ID
            # Grab article ID and add to list
            file_id = file_name.replace(self._dir[0],'').replace('.txt','').replace('\\article','')
            self.article_ids.append(file_id)

        # Loop over these id files
        for lbl in self.article_ids:
            # grab the .txt file with same id

            # add lbl-key and empty dict
            self.dataset_full[f'{lbl}'] = {}

            # find file id match
            for file_name in self.list_of_articles:
                # parse file name - remove 'article' and '.txt'
                if file_name.endswith(f'{lbl}.txt'):
                    # open file, create vector of contents
                    data_lines = []
                    with open(file_name, 'r', encoding="utf-8") as file:
                        for line in file:
                            data_lines.append(line.strip())

                    # add data_lines into dict
                    self.dataset_full[f'{lbl}']['data'] = data_lines
                    # print(lines)

            for file_name in self.list_of_labels:
                # parse file name
                if file_name.endswith(f'{lbl}.task-SLC.labels'):
                    # print(file_name)
                    with open(file_name, 'r', encoding='utf-8') as file:
                        lines = file.readlines()

                    # get 3rd col
                    labeled_list = [line.strip().split('\t')[2] for line in lines]

                    # add labels into dict
                    self.dataset_full[f'{lbl}']['labels'] = labeled_list


        # print(self.dataset_full)
        return self.dataset_full