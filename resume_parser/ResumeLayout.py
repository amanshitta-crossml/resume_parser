import pdfplumber
import os
import re
from fuzzywuzzy import process, fuzz
from itertools import tee
from math import ceil, floor
from resume_parser.layout_config import RESUME_HEADERS



def reformed_lines_dict_data(all_words_in_coords) -> dict:
    """
        Reformatting the OCR ouput in a line based format for identifying 
        words in same line struct.
    """
    try:
        lines = {}
        if all_words_in_coords:
            all_words_in_coords = [word for word in all_words_in_coords if word['text'].strip()]
            all_words_in_coords = sorted(all_words_in_coords, key=lambda x:(x['top'], x['x0']))
            curr_line = all_words_in_coords[0]['top']

            for index, word in enumerate(all_words_in_coords):
                if index <= len(all_words_in_coords)-1:
                    curr_top = all_words_in_coords[index]['top']
                    height = abs(word['bottom'] - word['top'])

                    if curr_top-(height//2) <= curr_line and curr_top+height <= curr_line+(height*1.5):
                        if lines:
                            lines[curr_line].append(word)
                        else:
                            curr_line = ceil(curr_top)
                            lines[curr_line] = [word]
                    else:
                        curr_line = ceil(curr_top)
                        lines.update({curr_line:[word]})
        if lines:
            lines = {curr_top:sorted(words, key=lambda x:x['x0']) for curr_top,words in sorted(lines.items())}
    except Exception as e:
        print(f"reformed_lines_dict_data :: Exception :: {e}")
    return lines

def words_concat_or_not(line, avg, processed = []):
    lines = []
    if not line: return line
    try:
        for idx in range(len(line)-1):
            if idx in processed: continue

            if line[idx+1]['x0']-line[idx]['x1'] <= avg:
                new_line = {'text': line[idx]['text']+" "+line[idx+1]['text'], 'x0':line[idx]['x0'], 'x1':line[idx+1]['x1'],  'top':line[idx]['top'], 'doctop':line[idx]['doctop'], 'bottom':line[idx+1]['bottom'], 'upright':line[idx+1]['upright'], 'direction':line[idx]['direction']}
                lines.append(new_line)
                processed.extend([idx,idx+1])
            else:
                if idx not in processed:
                    lines.append(line[idx])
                    processed.append(idx)
        else:
            if len(line)-1 not in processed:
                lines.append(line[-1])

    except Exception as e:
        print("concat_or_not :: Exception :: ", str(e))

    if not lines: lines = line
    return lines

def lines_concat_or_not(lines_list, avg_height_diff):
    if not lines_list: return lines_list
    formed_sentence = []
    processed = []
    for line_idx in range(0, len(lines_list)-1):
        if line_idx in processed: continue
        try:
            if  0 < (lines_list[line_idx+1]['top']-lines_list[line_idx]['bottom']) <= avg_height_diff and \
                lines_list[line_idx]['x0']<lines_list[line_idx+1]['x1']:
                new_sentence = {'text': lines_list[line_idx]['text']+" "+lines_list[line_idx+1]['text'],
                                'x0': min([i['x0'] for i in lines_list[line_idx:line_idx+2]]),
                                'top':min([i['top'] for i in lines_list[line_idx:line_idx+2]]),
                                'x1': max([i['x1'] for i in lines_list[line_idx:line_idx+2]]),
                                'doctop':min([i['doctop'] for i in lines_list[line_idx:line_idx+2]]),
                                'bottom': max([i['bottom'] for i in lines_list[line_idx:line_idx+2]]),
                                'upright':lines_list[line_idx+1]['upright'],
                                'direction': lines_list[line_idx]['direction']
                                }

                formed_sentence.append(new_sentence)
                processed.append(line_idx)
                processed.append(line_idx+1)
            else:
                if line_idx not in processed:
                    formed_sentence.append(lines_list[line_idx])
                    processed.append(line_idx)
        except Exception as e:
            print("lines_concat_or_not :: Exception :: ", str(e))

    else:
        if len(lines_list)-1 not in processed:
            formed_sentence.append(lines_list[-1])

    if not formed_sentence: formed_sentence = lines_list
    
    return formed_sentence

def lines_concat_or_not_par(lines_list, avg_height_diff):
    if not lines_list: return lines_list
    formed_sentence = []
    processed = []
    for line_idx in range(0, len(lines_list)-1):
        if line_idx in processed: continue
        try:
            if (lines_list[line_idx+1]['top']-lines_list[line_idx]['bottom']) <= avg_height_diff and True :
                # lines_list[line_idx]['x0']<lines_list[line_idx+1]['x1']:
            
                new_sentence = {'text': lines_list[line_idx]['text']+" "+lines_list[line_idx+1]['text'],
                                'x0': min([i['x0'] for i in lines_list[line_idx:line_idx+2]]),
                                'top':min([i['top'] for i in lines_list[line_idx:line_idx+2]]),
                                'x1': max([i['x1'] for i in lines_list[line_idx:line_idx+2]]),
                                'doctop':min([i['doctop'] for i in lines_list[line_idx:line_idx+2]]),
                                'bottom': max([i['bottom'] for i in lines_list[line_idx:line_idx+2]]),
                                'upright':lines_list[line_idx+1]['upright'],
                                'direction': lines_list[line_idx]['direction']
                                }
                formed_sentence.append(new_sentence)
                processed.append(line_idx)
                processed.append(line_idx+1)
            else:
                if line_idx not in processed:
                    formed_sentence.append(lines_list[line_idx])
                    processed.append(line_idx)
        except Exception as e:
            print("lines_concat_or_not :: Exception :: ", str(e))

    else:
        if len(lines_list)-1 not in processed:
            formed_sentence.append(lines_list[-1])

    if not formed_sentence: formed_sentence = lines_list
    
    return formed_sentence

def form_sentences(lines_list):
    lines_dict = reformed_lines_dict_data(lines_list)
    formed_line = []
    formed_sentences = []
    formed_paras = []

    for _, line in lines_dict.items():
        if len(line) <= 2 and line:
            if len(line)<=1:
                formed_line.extend(line)
            else:
                half_avg = sum([ceil(line[0]['x1'] - line[0]['x0'])//2, ceil(line[-1]['x1']-line[-1]['x0'])//2])/2
                if line[-1]['x0'] - line[0]['x1'] <= half_avg:
                    new_line = [{'text': line[0]['text']+" "+line[-1]['text'], 'x0':line[0]['x0'], 'x1':line[-1]['x1'],  'top':line[0]['top'], 'doctop':line[0]['doctop'], 'bottom':line[-1]['bottom'], 'upright':line[-1]['upright'], 'direction':line[0]['direction']}]
                    formed_line.extend(new_line)
                else:
                    formed_line.extend(line)
        else:
            avg_diff_list = [(line[idx+1]['x0'] - line[idx]['x1']) for idx in range(len(line)-1)]
            avg = ceil(sum(avg_diff_list)/len(avg_diff_list))
            new_line = []
            while True:
                new_line = words_concat_or_not(line, avg, processed = [])
                if line == new_line:
                    break
                line = new_line

            formed_line.extend(line)

    line_diff_list = [formed_line[idx+1]['top']-formed_line[idx]['bottom'] for idx in range(len(formed_line)-1) if (formed_line[idx+1]['top']-formed_line[idx]['bottom']>0 and  formed_line[idx+1]['top']-formed_line[idx]['bottom'] < 1.5*(formed_line[idx]['bottom'] - formed_line[idx]['top']))]
    if len(line_diff_list):
        avg_height_diff = ceil(sum(line_diff_list)/len(line_diff_list))

    else: avg_height_diff = 1

    lines = formed_line
    while True:
        formed_sentences = lines_concat_or_not(formed_line, avg_height_diff)
        if formed_line == formed_sentences:
            break
        formed_line = formed_sentences

    # form paragraphs
    try:
        line_diff_list = [formed_sentences[idx+1]['top']-formed_sentences[idx]['bottom'] for idx in range(len(formed_sentences)-1) if (formed_sentences[idx+1]['top']-formed_sentences[idx]['bottom']>0)]
        if len(line_diff_list):
            avg_height_diff = ceil(sum(line_diff_list)/len(line_diff_list))
        else:
            avg_height_diff = 1

        sen = formed_sentences
        
        while True:
            formed_paras = lines_concat_or_not_par(sen, avg_height_diff)
            if sen == formed_paras:
                break
            sen = formed_paras
    except:
        pass

    return lines, formed_sentences, formed_paras

def is_a_daterange(line):
    not_alpha_numeric = r'[^a-zA-Z\d]'
    number = r'(\d{2})'

    months_num = r'(01)|(02)|(03)|(04)|(05)|(06)|(07)|(08)|(09)|(10)|(11)|(12)'
    months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
    months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
    month = r'(' + months_num + r'|' + months_short + r'|' + months_long + r')'
    term = r'(summer|spring|winter|fall|)'
    regex_year = r'((20|19)(\d{2})|(\d{2}))'
    year = regex_year
    start_date = month + not_alpha_numeric + r"?" + year
    end_date = r'((' + number + r'?' + not_alpha_numeric + r"?" + month + not_alpha_numeric + r"?" + year + r')|(present|current|till date|today|now))'
    longer_year = r"((20|19)(\d{2}))"
    year_range = longer_year + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + r'(' + longer_year + r'|(present|current|till date|today|now))'
    term_range = r"(" + term + r"(" + year_range + r")" + r")"
    date_range = r"(" + start_date + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + end_date + r")|(" + year_range + r")|(" + term_range + r")|(" + months_num + r"\s?\/\s?" +regex_year + r")" + not_alpha_numeric + r"(" + months_num + r"\s?\/\s?" + r"\s?\/\s?" +regex_year + r")"

    regular_expression = re.compile(date_range, re.IGNORECASE)
    
    regex_result = re.search(regular_expression, line)
    
    if regex_result:
        return True


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

    def extract_work_sub_section(self, section, section_data, headers, sub_sections):

        sub_sections = {}

        section_top = [i for i in headers if i['text']==section][0]['bottom']
        lines = form_sentences(section_data[section])[0]
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

                sub_sections[idx] = (min([i['x0'] for i in section_data[section]]), 
                                                top,
                                                max([i['x1'] for i in section_data[section]]), 
                                                bottom)
            else:
                top = dates[-1]['bottom'] - date_dist
                if idx > 0 and dates[idx]['bottom']-date_dist < dates[idx-1]['bottom']:
                    top =  dates[-1]['bottom'] - (dates[-1]['bottom']-date_dist)-dates[idx]['bottom']
                else: top = dates[-1]['bottom'] - date_dist
                sub_sections[(len(dates)-1)] = (min([i['x0'] for i in section_data[section]]), 
                                                        top,
                                                        max([i['x1'] for i in section_data[section]]), 
                                                        max([i['bottom'] for i in section_data[section]]))
        return sub_sections

    def detect_subsection(self, section_data, headers):
        sub_sections = {}
        for section in section_data:
            sub_sections[section] = {}
            try:
                full_region = (min([i['x0'] for i in section_data[section]]), min([i['top'] for i in section_data[section]]), max([i['x1'] for i in section_data[section]]), max([i['bottom'] for i in section_data[section]]) )
            except: continue
            if 'experience' in section.lower():
                sub_sections[section].update(self.extract_work_sub_section(section, section_data, headers, sub_sections))
                if not sub_sections[section]:
                    sub_sections[section] = full_region
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
