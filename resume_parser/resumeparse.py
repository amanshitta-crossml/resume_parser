import nltk

import re
import os
from datetime import date
from fuzzywuzzy import process
import spacy

import nltk
import docx2txt
import pandas as pd
from tika import parser
import phonenumbers
import pdfplumber

import logging
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

import nltk
from stemming.porter2 import stem
from fuzzywuzzy import fuzz

# from resume_parser.ResumeLayout import ResumeLayoutParser, form_sentences
# from resume_parser.layout_config import RESUME_HEADERS

from ResumeLayout import ResumeLayoutParser, form_sentences
from layout_config import RESUME_HEADERS

from test import *

# bert
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def bert_organisation(line):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_results = nlp(line.strip())
    return ner_results

# load pre-trained model
base_path = os.path.dirname(__file__)


nlp = spacy.load('en_core_web_sm')
# custom_nlp2 = custom_nlp3 = nlp
custom_nlp2 = spacy.load(os.path.join(base_path,"degree","model"))
custom_nlp3 = spacy.load(os.path.join(base_path,"company_working","model"))

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

file = os.path.join(base_path,"titles_combined.txt")
file = open(file, "r", encoding='utf-8')
designation = [line.strip().lower() for line in file]
designitionmatcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(text) for text in designation if len(nlp.make_doc(text)) < 10]
designitionmatcher.add("Job title", None, *patterns)

file = os.path.join(base_path,"LINKEDIN_SKILLS_ORIGINAL.txt")
file = open(file, "r", encoding='utf-8')    
skill = [line.strip().lower() for line in file]
skillsmatcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(text) for text in skill if len(nlp.make_doc(text)) < 10]
skillsmatcher.add("Job title", None, *patterns)


nlp = spacy.load('en_core_web_sm')
class resumeparse(object):

    def convert_docx_to_txt(docx_file,docx_parser):
        """
            A utility function to convert a Microsoft docx files to raw text.

            This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
            :param docx_file: docx file with gets uploaded by the user
            :type docx_file: InMemoryUploadedFile
            :return: The text contents of the docx file
            :rtype: str
        """
        # try:
        #     text = docx2txt.process(docx_file)  # Extract text from docx file
        #     clean_text = text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        #     resume_lines = clean_text.splitlines()  # Split text blob into individual lines
        #     resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces

        #     return resume_lines
        # except KeyError:
        #     text = textract.process(docx_file)
        #     text = text.decode("utf-8")
        #     clean_text = text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
        #     resume_lines = clean_text.splitlines()  # Split text blob into individual lines
        #     resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]  # Remove empty strings and whitespaces
        #     return resume_lines
        try:
          if docx_parser == "tika":
            text = parser.from_file(docx_file, service='text')['content']
          elif docx_parser == "docx2txt":
            text = docx2txt.process(docx_file)
          else:
            logging.error('Choose docx_parser from tika or docx2txt :: ' + str(e)+' is not supported')
            return [], " "
        except RuntimeError as e:            
            logging.error('Error in tika installation:: ' + str(e))
            logging.error('--------------------------')
            logging.error('Install java for better result ')
            text = docx2txt.process(docx_file)
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
        try:
            clean_text = re.sub(r'\n+', '\n', text)
            clean_text = clean_text.replace("\r", "\n").replace("\t", " ")  # Normalize text blob
            resume_lines = clean_text.splitlines()  # Split text blob into individual lines
            resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if
                            line.strip()]  # Remove empty strings and whitespaces
            return resume_lines, text
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "

    def convert_pdf_to_txt(pdf_file):
        """
        A utility function to convert a machine-readable PDF to raw text.

        This code is largely borrowed from existing solutions, and does not match the style of the rest of this repo.
        :param input_pdf_path: Path to the .pdf file which should be converted
        :type input_pdf_path: str
        :return: The text contents of the pdf
        :rtype: str
        """
        # try:
            # PDFMiner boilerplate
            # pdf = pdfplumber.open(pdf_file)
            # full_string= ""
            # for page in pdf.pages:
            #   full_string += page.extract_text() + "\n"
            # pdf.close()

            
        try:
            raw_text = parser.from_file(pdf_file, service='text')['content']
        except RuntimeError as e:            
            logging.error('Error in tika installation:: ' + str(e))
            logging.error('--------------------------')
            logging.error('Install java for better result ')
            pdf = pdfplumber.open(pdf_file)
            raw_text= ""
            for page in pdf.pages:
              raw_text += page.extract_text() + "\n"
            pdf.close()                
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
        try:
            full_string = re.sub(r'\n+', '\n', raw_text)
            full_string = full_string.replace("\r", "\n")
            full_string = full_string.replace("\t", " ")

            # Remove awkward LaTeX bullet characters

            full_string = re.sub(r"\uf0b7", " ", full_string)
            full_string = re.sub(r"\(cid:\d{0,2}\)", " ", full_string)
            full_string = re.sub(r'• ', " ", full_string)

            # Split text blob into individual lines
            resume_lines = full_string.splitlines(True)

            # Remove empty strings and whitespaces
            resume_lines = [re.sub('\s+', ' ', line.strip()) for line in resume_lines if line.strip()]

            return resume_lines, raw_text
        except Exception as e:
            logging.error('Error in docx file:: ' + str(e))
            return [], " "
            
    def segment(res_segments, subsections):
        resume_segments = {}
        resume_subsections = {}
        # resume_segments = dict.fromkeys(RESUME_HEADERS.keys(), [])
        try:
            for header_section, keywords in RESUME_HEADERS.items():
                resume_segments[header_section] = resume_segments.get(header_section, [])
                # objecttive, skills, work_and_employment
                for pg, header_segments in res_segments.items():
                    # 0, {'SKILLS': [{word1}, {word2}]}
                    for header, header_words in header_segments.items():
                        # SKILLS, [{word1}, {word2}]}
                        for keyword in keywords:
                            # [carrier, goal, skills, projects ....]
                            if fuzz.ratio(header.lower(), keyword.lower()) >= 90:
                                resume_segments[header_section] = header_words
                                resume_subsections[header_section] = subsections[pg][header]

            for segment in resume_segments:
                if not resume_segments[segment] and segment in ['contact_info', 'objective']:
                    resume_segments[segment] = res_segments[0]['FREE_TEXT']

        except Exception as e:
            print("Exception :: segment :: ", str(e))
        return resume_segments, resume_subsections

    def calculate_experience(resume_text):
        
        #
        # def get_month_index(month):
        #   month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        #   return month_dict[month.lower()]
        # print(resume_text)
        # print("*"*100)
        def correct_year(result):
            if len(result) < 2:
                if int(result) > int(str(date.today().year)[-2:]):
                    result = str(int(str(date.today().year)[:-2]) - 1) + result
                else:
                    result = str(date.today().year)[:-2] + result
            return result

        # try:
        experience = 0
        start_month = -1
        start_year = -1
        end_month = -1
        end_year = -1

        not_alpha_numeric = r'[^a-zA-Z\d]'
        number = r'(\d{2})'

        months_num = r'(01)|(02)|(03)|(04)|(05)|(06)|(07)|(08)|(09)|(10)|(11)|(12)'
        months_short = r'(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)'
        months_long = r'(january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december)'
        month = r'(' + months_num + r'|' + months_short + r'|' + months_long + r')'
        regex_year = r'((20|19)(\d{2})|(\d{2}))'
        year = regex_year
        start_date = month + not_alpha_numeric + r"?" + year
        
        # end_date = r'((' + number + r'?' + not_alpha_numeric + r"?" + number + not_alpha_numeric + r"?" + year + r')|(present|current))'
        end_date = r'((' + number + r'?' + not_alpha_numeric + r"?" + month + not_alpha_numeric + r"?" + year + r')|(present|current|till date|today))'
        longer_year = r"((20|19)(\d{2}))"
        year_range = longer_year + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + r'(' + longer_year + r'|(present|current|till date|today))'
        date_range = r"(" + start_date + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))" + end_date + r")|(" + year_range + r")"

        
        regular_expression = re.compile(date_range, re.IGNORECASE)
        
        regex_result = re.search(regular_expression, resume_text)
        
        while regex_result:

          try:
            date_range = regex_result.group()
            # print(date_range)
            # print("*"*100)
            try:
              
                year_range_find = re.compile(year_range, re.IGNORECASE)
                year_range_find = re.search(year_range_find, date_range)
                # print("year_range_find",year_range_find.group())
                                
                # replace = re.compile(r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))", re.IGNORECASE)
                replace = re.compile(r"((\s*to\s*)|" + not_alpha_numeric + r"{1,4})", re.IGNORECASE)
                replace = re.search(replace, year_range_find.group().strip())
                # print(replace.group())
                # print(year_range_find.group().strip().split(replace.group()))
                start_year_result, end_year_result = year_range_find.group().strip().split(replace.group())
                # print(start_year_result, end_year_result)
                # print("*"*100)
                start_year_result = int(correct_year(start_year_result))
                if (end_year_result.lower().find('present') != -1 or 
                    end_year_result.lower().find('current') != -1 or 
                    end_year_result.lower().find('till date') != -1 or 
                    end_year_result.lower().find('today') != -1): 
                    end_month = date.today().month  # current month
                    end_year_result = date.today().year  # current year
                else:
                    end_year_result = int(correct_year(end_year_result))


            except Exception as e:
                # logging.error(str(e))
                start_date_find = re.compile(start_date, re.IGNORECASE)
                start_date_find = re.search(start_date_find, date_range)

                non_alpha = re.compile(not_alpha_numeric, re.IGNORECASE)
                non_alpha_find = re.search(non_alpha, start_date_find.group().strip())

                replace = re.compile(start_date + r"(" + not_alpha_numeric + r"{1,4}|(\s*to\s*))", re.IGNORECASE)
                replace = re.search(replace, date_range)
                date_range = date_range[replace.end():]
        
                start_year_result = start_date_find.group().strip().split(non_alpha_find.group())[-1]

                # if len(start_year_result)<2:
                #   if int(start_year_result) > int(str(date.today().year)[-2:]):
                #     start_year_result = str(int(str(date.today().year)[:-2]) - 1 )+start_year_result
                #   else:
                #     start_year_result = str(date.today().year)[:-2]+start_year_result
                # start_year_result = int(start_year_result)
                start_year_result = int(correct_year(start_year_result))

                if date_range.lower().find('present') != -1 or date_range.lower().find('current') != -1:
                    end_month = date.today().month  # current month
                    end_year_result = date.today().year  # current year
                else:
                    end_date_find = re.compile(end_date, re.IGNORECASE)
                    end_date_find = re.search(end_date_find, date_range)

                    end_year_result = end_date_find.group().strip().split(non_alpha_find.group())[-1]

                    # if len(end_year_result)<2:
                    #   if int(end_year_result) > int(str(date.today().year)[-2:]):
                    #     end_year_result = str(int(str(date.today().year)[:-2]) - 1 )+end_year_result
                    #   else:
                    #     end_year_result = str(date.today().year)[:-2]+end_year_result
                    # end_year_result = int(end_year_result)
                    try:
                      end_year_result = int(correct_year(end_year_result))
                    except Exception as e:
                      logging.error(str(e))
                      end_year_result = int(re.search("\d+",correct_year(end_year_result)).group())

            if (start_year == -1) or (start_year_result <= start_year):
                start_year = start_year_result
            if (end_year == -1) or (end_year_result >= end_year):
                end_year = end_year_result

            resume_text = resume_text[regex_result.end():].strip()
            regex_result = re.search(regular_expression, resume_text)
          except Exception as e:
            logging.error(str(e))
            resume_text = resume_text[regex_result.end():].strip()
            regex_result = re.search(regular_expression, resume_text)
            
        return end_year - start_year  # Use the obtained month attribute

    def get_experience(resume_segments):
        total_exp = 0
        if len(resume_segments['work_and_employment']):
            text = ""
            for word in resume_segments['work_and_employment']['sentences']:
                text += word
            total_exp = resumeparse.calculate_experience(text)
            return total_exp, text
        return total_exp, " "

    def find_phone(text):
        try:
            return list(iter(phonenumbers.PhoneNumberMatcher(text, None)))[0].raw_string
        except:
            try:
                return re.search(
                    r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                    text).group()
            except:
                return ""

    def extract_email(text):
        email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None

    def extract_name(resume_text):
        nlp_text = nlp(resume_text)

        # First name and Last name are always Proper Nouns
        # pattern_FML = [{'POS': 'PROPN', 'ENT_TYPE': 'PERSON', 'OP': '+'}]

        pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
        matcher.add('NAME', None, pattern)

        matches = matcher(nlp_text)

        for match_id, start, end in matches:
            span = nlp_text[start:end]
            return span.text
        return ""

    def extract_location(resume_text):
        
        location = ''
        try:         
            parsed_addr = nlp(resume_text)
            # breakpoint()
            for ent in parsed_addr.ents:
                if ent.label_ == 'GPE':
                    location = ent.text
                    break
        except Exception as e:
            print("attributes_extraction (libpostal): error: ", e)
        return location

    def extract_university(text, file):
        df = pd.read_csv(file, header=None)
        universities = [i.lower() for i in df[1]]
        college_name = []
        listex = universities
        listsearch = [text.lower()]

        for i in range(len(listex)):
            for ii in range(len(listsearch)):
                
                if re.findall(listex[i], re.sub(' +', ' ', listsearch[ii])):
                
                    college_name.append(listex[i])
        
        return college_name

    def job_designition(text):
        job_titles = []
        
        __nlp = nlp(text.lower())
        
        matches = designitionmatcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            job_titles.append(span.text)
        return job_titles

    def get_degree(text):
        degree_df = pd.read_csv(os.path.join(base_path, 'degree.csv'))
        extracted = process.extractOne(text, degree_df['degree'].to_list(), scorer=fuzz.ratio)
        if extracted[1]>95:
            return extracted[0]

    def get_company_working(text):
        doc = custom_nlp3(text)
        degree = []

        degree = [ent.text.replace("\n", " ") for ent in list(doc.ents)]
        return list(dict.fromkeys(degree).keys())
    
    def extract_skills(text):

        skills = []

        __nlp = nlp(text.lower())
        # Only run nlp.make_doc to speed things up

        matches = skillsmatcher(__nlp)
        for match_id, start, end in matches:
            span = __nlp[start:end]
            skills.append(span.text)
        skills = list(set(skills))
        return skills

    def parse_org_name(org):
        org_list = [(i['word']) for i in org if 'ORG' in i['entity']]

        org_name = ''
        for org_token in org_list:
            if org_token.startswith('##'):
                org_token = org_token.lstrip('#')
                org_name+=org_token
            else:
                org_name=org_name+' '+org_token

        return org_name.strip()

    def extract_work_employment(experience_subsections):
        out = []
        # Extract Organisation
        for idx, subsection in experience_subsections.items():
            subsection_lines = form_sentences(subsection)[0]
            temp = {}
            extra_text = []
            for line in subsection_lines:
                if not temp.get('organisation_name'):
                    org = bert_organisation(line['text'])
                    if org:
                        org_name = resumeparse.parse_org_name(org)
                        if org_name :
                            temp = {"organisation_name": org_name}
                            continue
                
                exp = resumeparse.calculate_experience(line['text'])
                if exp:
                    temp.update({"experience": exp})
                    continue
                else:
                    extra_text.append(line)

            temp.update({'job_role': [i['text'] for i in sorted(extra_text, key= lambda x: (x['top'], x['x1']))]})
            out.append(temp)
        return out

    def read_file(file,docx_parser = "tika"):
        """
        file : Give path of resume file
        docx_parser : Enter docx2txt or tika, by default is tika
        """

        file = os.path.join(file)

        resume = ResumeLayoutParser(file)
        
        doc_headers, res_segments, subsections = resume.process_resume()

        resume_lines = []
        for page, cols in res_segments.items():
            for _, col in cols.items():
                resume_lines += [i['text'] for i in col]

        resume_segments, resume_subsections = resumeparse.segment(res_segments, subsections)

        for key, segment in resume_segments.items():
            lines, sentences, _ = form_sentences(segment)
            resume_segments.update({key: {'lines': [i['text'] for i in lines], 'sentences': [i['text'] for i in sentences]}})

        email = resumeparse.extract_email(' '.join(resume_segments['contact_info']['lines']))

        phone = resumeparse.find_phone(' '.join(resume_segments['contact_info']['lines']))

        name = resumeparse.extract_name(" ".join(resume_segments['contact_info']['lines']))

        location = resumeparse.extract_location(" ".join(resume_segments['contact_info']['lines']))

        total_exp, text = resumeparse.get_experience(resume_segments)

        university = resumeparse.extract_university(' '.join(resume_segments['education_and_training']['lines']), os.path.join(base_path,'world-universities.csv'))

        skills = []

        experience_subsections = resumeparse.extract_work_employment(resume_subsections['work_and_employment'])

        if len(skills) == 0 and resume_segments['skills']:
            skills = resumeparse.extract_skills(' '.join(resume_segments['skills']['sentences']))
        skills = list(dict.fromkeys(skills).keys())
        return {
            "email": email,
            "phone": phone,
            "name": name,
            "total_exp": total_exp,
            "location": location,
            "university": university,
            "certificate": resume_segments['certificate']['sentences'],
            'projects': resume_segments['projects']['sentences'],
            "languages": resume_segments['language']['lines'],
            "interests": resume_segments['interests']['sentences'],
            "education": resume_segments['education_and_training']['lines'],
            "skills": skills,
            "experience": experience_subsections
        }
