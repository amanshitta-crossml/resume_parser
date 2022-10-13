import re
from math import ceil

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
                new_line = {'text': line[idx]['text']+" "+line[idx+1]['text'], 'x0':line[idx]['x0'], 'x1':line[idx+1]['x1'],  'top':line[idx]['top'], 'doctop':line[idx]['doctop'], 'bottom':line[idx+1]['bottom'], 'upright':line[idx+1]['upright'], 'direction':line[idx]['direction'], 'stroking_color':line[idx]['stroking_color'], 'fontname': line[idx].get('fontname', ''), 'size': line[idx]['size']}
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
                                'direction': lines_list[line_idx]['direction'],
                                'stroking_color':lines_list[line_idx]['stroking_color'], 
                                'fontname': lines_list[line_idx].get('fontname', ''), 
                                'size': lines_list[line_idx].get('size', '')
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
                                'direction': lines_list[line_idx]['direction'],
                                'stroking_color':lines_list[line_idx]['stroking_color'], 
                                'fontname': lines_list[line_idx].get('fontname', ''), 
                                'size': lines_list[line_idx].get('size', '')
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
                    new_line = [{'text': line[0]['text']+" "+line[-1]['text'], 'x0':line[0]['x0'], 'x1':line[-1]['x1'],  'top':line[0]['top'], 'doctop':line[0]['doctop'], 'bottom':line[-1]['bottom'], 'upright':line[-1]['upright'], 'direction':line[0]['direction'], 'stroking_color':line[0]['stroking_color'], 'fontname': line[0].get('fontname', '')}]
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

    # for lines sentence formation
    line_diff_list = [formed_line[idx+1]['top']-formed_line[idx]['bottom'] for idx in range(len(formed_line)-1) if (formed_line[idx+1]['top']-formed_line[idx]['bottom']>0 and  formed_line[idx+1]['top']-formed_line[idx]['bottom'] < 1.5*(formed_line[idx]['bottom'] - formed_line[idx]['top']))]

    # for paraghrphs formation
    par_line_diff_list = [formed_line[idx+1]['top']-formed_line[idx]['bottom'] for idx in range(len(formed_line)-1) if (formed_line[idx+1]['top']-formed_line[idx]['bottom']>0)]

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
        if len(par_line_diff_list):
            avg_height_diff = (sum(par_line_diff_list)/len(par_line_diff_list))
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

def is_a_daterange(line, extract_range=False):
    not_alpha_numeric = r'[^a-zA-Z\d]'
    number = r'(\d{2})'

    months_num = r'(01)|(02)|(03)|(04)|(05)|(06)|(07)|(08)|(09)|(10)|(11)|(12)'
    # months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
    MONTHS_PATTERN = r"january|february|march|april|may|june|july|august|september|october|november|december|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december|jan\.?|ene\.?|feb\.?|mar\.?|apr\.?|abr\.?|may\.?|maj\.?|jun\.?|jul\.?|aug\.?|ago\.?|sep\.?|sept\.?|oct\.?|okt\.?|nov\.?|dec\.?|dic\.?"
    # months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
    month = r'(' + months_num + r'|' + MONTHS_PATTERN + r')'
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
        if not extract_range:
            return True
        else:
            return True, regex_result.group()
    


def group_lines(words):
    word_dict = {}
    sentence_dict = {}
    for word in words:
        key = (word['size'],word['fontname'], tuple(word['stroking_color']) if isinstance(word['stroking_color'], list) else word['stroking_color'], word['isUpper'])
        if key in word_dict:
            word_dict[key].append(word)
        else:
            word_dict[key] = [word]
    return word_dict
  
def designation_fallback(parsed, subsection_lines):
    possible_designations = []
    lines = [line for idx, line in enumerate(subsection_lines) if idx not in parsed]
    for word in lines:
        word['size'] = word['bottom'] - word['top']
        word['isUpper'] = word['text'].isupper()

    grouped_lines = group_lines(lines)
    max_key = max(grouped_lines.keys(),key=lambda x:x[0])
    for key, value in grouped_lines.items():
        if key[0] == max_key[0] or 'bold' in key[1].lower():
            vals = []
            for val in value:
                try:
                    ord(val['text'])
                except:
                    if len(val['text'].split()) < 6:
                        vals.append(val)
            if vals:
                possible_designations.extend(vals)
    try:
        possible_designations = sorted(possible_designations, key = lambda x: x['isUpper'], reverse=True)
    except:
        # do nothing
        pass

    return possible_designations[0]['text'] if  possible_designations else ''


