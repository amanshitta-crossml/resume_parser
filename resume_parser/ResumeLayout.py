import enum
import pdfplumber
from helper import *
from fuzzywuzzy import fuzz
from resume_parser.layout_config import *
# from layout_config import RESUME_HEADERS, EDU_RESERVED_WORDS

class ResumeLayoutParser():

    def __init__(self, path):
        self.pages_shape = {}
        self.headers = {}
        self.bold_letters = {}
        self.columns = {}
        self.chars = {}
        self.words = {}
        self.column_data = {}
        self.formed_headers = {}
        self.tables = {}
        self.tables_text = {}
        self.document = path
        self.get_basic_data()

    def get_basic_data(self):
        """
            Find bold characters and form words.
        """
        with pdfplumber.open(self.document) as pdf:
            pages = pdf.pages
            for idx, page in enumerate(pages):
                self.pages_shape[idx] = {'width': page.width, 'height': page.height}
                self.chars[idx] = page.chars
                self.words[idx] = page.extract_words(use_text_flow=True, extra_attrs=['size', 'fontname', 'stroking_color', 'non_stroking_color'], x_tolerance=2.5)
                self.tables[idx] = page.extract_tables()
                self.tables_text[idx] = page.extract_text()
                self.words[idx] = self.detect_capitalised(self.words[idx])
                self.words[idx] = self.optimise_word_size(self.words[idx])

    def detect_capitalised(self, words):
        for word in words:
            if word['text'].isupper():
                word['isUpper'] = True
            else:
                word['isUpper'] = False
        return words

    def optimise_word_size(self, words):
        for word in words:
            word['size'] = round(word['size'], 4)
        return words

    def detect_line_height(self,words):
        for word in words:
            word['size'] = word['bottom'] - word['top']
        return words

    @staticmethod
    def get_valid_headers(header_list):
        new_list = []
        try:
            count = 0
            for word in header_list:
                for word_list in RESUME_HEADERS.values():
                    for wordx in word_list:
                        if fuzz.ratio(wordx.lower(), word['text'].lower())>85:
                            new_list.append(word)
                            break
        except Exception as e:
            print("get_match_score :: Exception :: ", str(e))
        return new_list

    @staticmethod
    def find_possble_header(words):
        word_dict = {}
        sentence_dict = {}
        for word in words:
            key = (word['size'],word['fontname'], tuple(word['stroking_color']) if isinstance(word['stroking_color'], list) else word['stroking_color'], word['isUpper'])
            # print('word_dict',word_dict)
            if key in word_dict:
                word_dict[key].append(word)
            else:
                word_dict[key] = [word]



        for attribute_key, word_data in word_dict.items():
            sentence_dict[attribute_key] = form_sentences(word_data)[1]
        return word_dict, sentence_dict

    def match_header(self, sentence_dict):
        matched_header = []
        for attribute_key, sentence in sentence_dict.items():
            # print(attribute_key, [o['text'] for o in sentence])
            matched_header.extend(self.get_valid_headers(sentence))
        matched_header = [header for header in matched_header if header]
        return matched_header

    def detect_column(self, matched_header_list):
        col_data = {'left':[], 'right': []}
        for header in matched_header_list:
            if header['x0'] < self.img_dim[0]/4:
                col_data['left'].append(header)
            else:
                col_data['right'].append(header)
        return col_data

    def detect_section(self, page, column_data):
        sections_data = {}
        sections_data['FREE_TEXT'] = []
        headers = []
        
        for column_name, column in column_data.items():
            sorted_columns = sorted(column, key=lambda x:x['top'])
            headers.extend(sorted_columns)
            header_top_list = [col['top'] for col in sorted_columns]
            left = min([i['x0'] for i in sorted(column, key=lambda x:x['x0'])])
            right = self.img_dim[1]

            if len(column)>1:
                header_pairs = list(zip(header_top_list, header_top_list[1:] + [self.img_dim[1]]))
                sorted_columns_right = sorted(column_data.get('right', []), key=lambda x:x['x0'])
                if sorted_columns_right:
                    right = sorted_columns_right[0]['x0'] - 20
                else:
                    left = 0
                if column_name == 'right':
                    right = self.img_dim[0]
                    left = sorted_columns_right[0]['x0']
                    if not column_data.get('left', []):
                        left = 0 
            else:
                header_pairs = list(zip(header_top_list, [self.img_dim[1]]))            

            for pair_index, header_pair in enumerate(header_pairs):
                header_key = sorted_columns[pair_index]['text']
                sections_data[header_key] = []                

                for word in self.words[page]:
                    if word['top'] > header_pair[0] and\
                        word['bottom'] < header_pair[1] and\
                        word['x1'] > left and\
                        word['x0'] < right:                        
                        sections_data[header_key].append(word)

                if sections_data[header_key]:
                    sections_data[header_key] = sorted(sections_data[header_key], key=lambda x: (x['top'], x['x0']))

            free_div_cords = [0,0,right,min(header_top_list)]
            for word in self.words[page]:
                if word['top'] > free_div_cords[1] and\
                    word['bottom'] < free_div_cords[3] and\
                    word['x1'] < free_div_cords[2] and\
                    word['x0'] > free_div_cords[0]:                        
                    sections_data['FREE_TEXT'].append(word)

            if sections_data['FREE_TEXT']:
                sections_data['FREE_TEXT'] = sorted(sections_data['FREE_TEXT'], key=lambda x: (x['top'], x['x0']))
        return sections_data, headers

    @staticmethod
    def detect_date_range(sentence):
        dates = []
        processed_date_top=[]
        for line in sentence:
            # print(line)
            if is_a_daterange(line['text']) and line['top'] not in processed_date_top:
                dates.append(line)
                processed_date_top.append(line['top'])


        return dates

    def extract_education_subsections(self, section_words):
        sub_sections = {}
        try:
            if len(section_words) > 1:
                _, _, sections = form_sentences(section_words)
                for idx, sections in enumerate(sections):
                    sub_sections[idx] = (sections['x0'], sections['top'],sections['x1'], sections['bottom'])
            else:
                sub_sections[0] = (min([i['x0'] for i in section_words]), 
                                    min([i['top'] for i in section_words]),
                                    max([i['x1'] for i in section_words]), 
                                    max([i['bottom'] for i in section_words]))
        except Exception as e:
            print("extract_education_subsections :: Excption :: ", str(e))
        return sub_sections

    def extract_work_subsection(self, section_header, section_words, headers):

        sub_sections = {}

        section_top = [i for i in headers if i['text']==section_header][0]['bottom']
        lines = form_sentences(section_words)[0]
        dates = (self.detect_date_range(lines))
        # print('dates', dates)
        if dates and len(dates)>1:
            first_date = sorted(dates, key=lambda x: x['top'])[0]
            date_dist = first_date['top']-section_top
            for idx in range(len(dates)-1):
                top = dates[idx]['bottom'] - date_dist
                bottom = dates[idx+1]['bottom'] - date_dist

                if idx > 0 and dates[idx]['bottom']-date_dist < dates[idx-1]['bottom']:
                    top =  dates[idx]['bottom'] - (dates[idx]['bottom']-date_dist)-dates[idx-1]['bottom']
                    bottom = dates[idx+1]['bottom'] - (dates[idx]['bottom']-date_dist)-dates[idx-1]['bottom']

                sub_sections[idx] = (min([i['x0'] for i in section_words]), 
                                                top,
                                                max([i['x1'] for i in section_words]), 
                                                bottom)
            else:
                top = dates[-1]['bottom'] - date_dist
                if idx > 0 and dates[idx]['bottom']-date_dist < dates[idx-1]['bottom']:
                    top =  dates[-1]['bottom'] - (dates[-1]['bottom']-date_dist)-dates[idx]['bottom']
                else: top = dates[-1]['bottom'] - date_dist
                sub_sections[(len(dates)-1)] = (min([i['x0'] for i in section_words]), 
                                                        top,
                                                        max([i['x1'] for i in section_words]), 
                                                        max([i['bottom'] for i in section_words]))
        else:
            sub_sections[0] = (min([i['x0'] for i in section_words]), 
                                min([i['top'] for i in section_words]),
                                max([i['x1'] for i in section_words]), 
                                max([i['bottom'] for i in section_words]))
        return sub_sections

    def detect_subsection(self, section_data, headers):
        sub_sections = {}
        for section in section_data:
            sub_sections[section] = {}
            try:
                full_region = (min([i['x0'] for i in section_data[section]]), min([i['top'] for i in section_data[section]]), max([i['x1'] for i in section_data[section]]), max([i['bottom'] for i in section_data[section]]) )
            except: continue
            # get work experience subsections
            if 'experience' in section.lower():
                sub_sections[section].update(self.extract_work_subsection(section, section_data[section], headers))
                if not sub_sections[section]:
                    sub_sections[section] = full_region
            
            # get work experience subsections
            elif 'education' in section.lower():
                sub_sections[section].update(self.extract_education_subsections(section_data[section]))
            else:
                sub_sections[section][0] = full_region

        return sub_sections

    def get_subsection_words(self, page, section_data, headers):
        subsec_words = {}
        try:
            subsec_coords = self.detect_subsection(section_data, headers)
            for header, cords in subsec_coords.items():
                subsec_words[header] = {}
                for idx, cord in cords.items():
                    subsec_words[header][idx] = []
                    for word in self.words[page]:
                        if word['x0'] >= cord[0] and word['top']>=cord[1] and word['x1'] <= cord[2] and word['bottom'] <= cord[3]:
                            subsec_words[header][idx].append(word)
        except Exception as e:
            print("get_subsection_words :: Exception :: ", str(e))                
        
        return subsec_words

    def detect_layout(self):
        self.doc_headers = {}
        self.res_segments = {}
        self.subsection_words = {}

        for page in self.words:
            self.res_segments[page] = {}
            try:
                self.img_dim = (self.pages_shape[page]['width'], self.pages_shape[page]['height'])
                words = self.words[page]
                word_dict, sentence_dict = self.find_possble_header(words)
                matched_header_list = self.match_header(sentence_dict)
                if matched_header_list:
                    self.column_data[page] = self.detect_column(matched_header_list)
                    section_data, headers = self.detect_section(page, self.column_data[page])
                    subsec_words = self.get_subsection_words(page, section_data, headers)
                    self.res_segments[page] = section_data
                    self.doc_headers[page] = headers
                    self.subsection_words[page] = subsec_words

            except Exception as e:
                print("detect_layout :: Exception :: ", str(e))
                
    def process_resume(self):
            self.detect_layout()
            return self.doc_headers, self.res_segments, self.subsection_words
