import os
import zipfile 
import re
import multiprocessing
import requests
import time
import logging
import copy

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import Dict, Union, Optional
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from lm_datasets.datasets.base import BaseDataset, Availability, MILLION, License

logger = logging.getLogger(__name__)



class DELawsDataset(BaseDataset):
    """
    Class to download, process and yield a dataset of german law.
    Parts of code adapted from:
    @bundestag
    https://github.com/bundestag/de_laws_to_json#Apache-2.0-1-ov-file
    """

    DATASET_ID = "delaws"           # TODO: Check Naming conventions
    TITLE = "DELaws"
    DESCRIPTION = (
        "Crawled law documents from https://www.gesetze-im-internet.de"
    )
    LANGUAGES = ['de']
    AVAILABILITY = Availability.DIRECT_DOWNLOAD
    WEB_CRAWLED = True
    LICENSE = License(
        "public domain",
        url="https://www.gesetze-im-internet.de/urhg/__5.html",
        commercial_use=True,
        sharealike=False,
        research_use=True,
    )
    TOKENS = 1.5 * MILLION

    def custom_request(self, url, max_retries=5):
        """
        Common requests with a timer for 1s between calls. In case of timeout
        uses an increasing sleep period.
        """

        for i in range(max_retries):
            try:
                response = requests.get(url)
                # time.sleep(1)
                return response
            except requests.exceptions.ConnectionError:
                logger.debug(f"Request timed out. Retrying (attempt {i + 1}/{max_retries})...")
                time.sleep(2**(i+1))
        logger.debug(f"Max retries reached. File was not downloaded: {url}")


    def process_law(self,law):
        """
        Function to process each item from the item array. It does the following for each item.
        """
        # Download the zip file
        
        item_response = self.custom_request(law['link'])

        zip_name = re.sub(r'\W+', '', law['link']) + '.zip'
        zip_path = os.path.join(law['output_dir'], zip_name)
        with open(zip_path, 'wb') as file_driver:
            for chunk in item_response.iter_content(chunk_size=128):
                file_driver.write(chunk)
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.xml'):
                    zip_ref.extract(file_name, law['output_dir'])
        # Remove the zip file
        os.remove(zip_path)
        return 1

    def num_tokens_from_string(string: str) -> int:
        """
        Function to count the number of tokens in a string
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def convert_xml_to_dict(self, element, expected_type: Optional[type] = None) -> Union[str, Dict]:
        """
        Function to recursively convert xml element and its children into dictionary.
        """
        if element.string:
            return element.string
        else:
            children_dict = {}
            for child in element.contents:
                if child.name:
                    if child.name in children_dict:
                        if isinstance(children_dict[child.name], list):
                            children_dict[child.name].append(self.convert_xml_to_dict(child))
                        else:
                            children_dict[child.name] = [children_dict[child.name], self.convert_xml_to_dict(child)]
                    else:
                        children_dict[child.name] = self.convert_xml_to_dict(child)
            # The final return should be a dict (when this function is not called by itself)
            if expected_type is not None and not isinstance(children_dict, expected_type):
                raise ValueError(f"Expected {expected_type} but got {type(children_dict)}")
            return children_dict

    def download(self):
        """
        List all download URLs and download the zip file with all laws and run
        process_law for each one, saving the XML files.
        """

        # Download the XML file
        response = self.custom_request('https://www.gesetze-im-internet.de/gii-toc.xml')

        # Parse the XML from the response text
        root = ET.fromstring(response.content)

        # Create an array of dictionaries
        item_array = []
        for item in root.findall('.//item'):
            title = item.find('title').text
            link = item.find('link').text
            item_dict = {'title': title, 'link': link, 'output_dir': self.output_dir}
            item_array.append(item_dict)

        # Create directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Set the number of items to process
        num_items_to_process = len(item_array)  # change this to control how many items to process
        logger.info(f"Processing {num_items_to_process} items out of {len(item_array)} total items")

        # Initialize a Pool with the number of workers defined in the BaseClass
        pool = multiprocessing.Pool(processes=self.workers)
        logger.info(f"Using {self.workers} cores/processes in parallel")

        # Use Pool's map function to process the items in parallel
        with tqdm(total=num_items_to_process, desc="Processing files", dynamic_ncols=True) as pbar:
            for _ in pool.imap_unordered(self.process_law, item_array[:num_items_to_process]):
                pbar.update()
    
    def process_text(self, filename):
        """
        For each XML file
        """
        # simple output dict for storing norms
        output = {'norms':[]}
        # Read file
        file_path = os.path.join(self.output_dir, filename)

        with open(file_path, encoding="utf8") as file:
            file_content = file.read()

            # Parse XML with BeautifulSoup
            soup = BeautifulSoup(file_content, "lxml-xml")
            
            # The following code belongs to the bundestag repository.
            # I need to get only the title and norm content, therefore the xml
            # extraction is maintened as is, but we keep the relevant elements

            """
            Get the norms of the law
            """
            for law in soup.find_all('norm'):
                this_norm = {
                    'meta': {},
                    'paragraphs': []
                }

                """
                Norm Metadata
                """
                this_metadaten = self.convert_xml_to_dict(law.find('metadaten'), dict)

                # For now, Only process norms that start with §, Art, Artikel (everything else is e.g. Inhaltsverzeichnis, Anlage) (TODO)
                pattern_norm = r'(§+|Art|Artikel)\.?\s*'


                if isinstance(this_metadaten, dict) and this_metadaten.get('enbez') and re.match(pattern_norm, this_metadaten['enbez']):
                    this_norm['meta'] = {
                        'norm_id': this_metadaten['enbez'],
                        'title': ''
                    }
                    try:
                        this_norm['meta']['title'] = law.find('metadaten').titel.text
                    except AttributeError:
                        pass

                    # Some laws have a "Gliederung", e.g. Art I, Art II. This would lead to duplicate titles if we ignore it
                    # With this, it will look like this: Art I §1, Art II §1
                    if this_metadaten.get('gliederungseinheit') and this_metadaten.get('gliederungseinheit').get('gliederungsbez'):
                        this_norm['meta']['norm_id'] = this_metadaten['gliederungseinheit']['gliederungsbez'] + ' ' + this_norm['meta']['norm_id']

                    """
                    Norm Content
                    """
                    if (law.find('textdaten') and law.find('textdaten').find('text') and law.find('textdaten').find('text').find('Content')):

                        """
                        Norm Content - P Tag (Absätze)
                        Wa want to put all Absätze in an array of paragraphs with their paragraph number.

                        Some paragraphs are numbered at the beginning of each paragraph, e.g. "(1) Die...".
                        Of those, sometimes a new P tag starts without a new number meaning it belongs to the previous paragraph.
                        For this logic, we need p_is_numbered so that we now that the paragraphs in the norm are numbered.

                        If a paragraph is not numbered, we will count ourselves with p_i.
                        """
                        this_content = law.find('textdaten').find('text').find('Content')
                        whitespace_pattern = r"\n\s+\n"  # Some paragraphs have a lot of whitespace which we will remove.
                        p_i = 0
                        p_is_numbered = False
                        for P in this_content.find_all('P', recursive=False):
                            # recursive=False so that we only get direct children (and e.g. not nested Ps such as in 'Revision' tags)
                            # Examples for laws with Revision tags: e.g. kstg § 34. Lambda e.g. bmelddav §5
                            p_i += 1
                            number = p_i
                            number_missing = False

                            # We want to check if the P tag has numbering in the beginning [(1) or 1]
                            # so that we can use it as it is more reliable then counting ourselves.
                            # However, we need to remove DL, Revision and table tags which sometimes also start with nubmers.
                            P_copy = copy.deepcopy(P)
                            for tag in P_copy.find_all(['DL', 'Revision', lambda t: t.name == 'entry' and t.get('colname') == 'col1']):
                                tag.decompose()
                            P_split = P_copy.text.split()  # We split the text at the first whitespace, leaving us with the first word.
                            # Now, we can identify the right number for the paragraph
                            if P_split:
                                first_part = P_split[0]
                                pattern_number = r"\b\d+[a-zA-Z]?\b"
                                # If the regex matches, we have a number (with optionally one letter, such as 1b)
                                match = re.search(pattern_number, first_part)
                                if match:  # If a match was found
                                    number = match.group()  # Get the matched string
                                    number = re.sub(r'\W+', '', number)  # Remove non-word characters (not a letter, digit)
                                    p_is_numbered = True  # We now know that this norm has numbered paragraphs.

                                    # Some laws have errors, e.g. BJNR048500995 § 6 has two (2).
                                    # Therefore we need to check if we would add a duplicate. (TODO - optimize part)
                                    for paragraph in this_norm['paragraphs']:
                                        number = str(number)
                                        if str(paragraph['meta']['paragraph_id']) == number:
                                            if bool(re.match('^\d+$', number)):
                                                number = int(number)
                                                number += 1
                                            else:
                                                number = str(number) + "_"
                                            break  # For now we're not correcting the wrong (2) at the beginning

                                # If we have not found a match, but previously did, this P tag continues the previous paragraph.
                                elif p_is_numbered:
                                    number_missing = True
                                    number = p_i-1
                                # If no match was found, the P has unumbered paragraphs and we will count ourselves.
                                else:
                                    number = p_i

                            # Remove all SUP tags for now. Those are the little numbers in the text that refer to the sentence number (TODO).
                            for sup in P('SUP'):
                                sup.extract()

                            # This is our paragraph object that we will push to the paragraphs array.
                            # This configuration of get_text() strips all text of leading and ending whitespace
                            # and then puts all text togther separated by a whitespace.
                            p_obj = {
                                'meta': {
                                    'paragraph_id': str(number),
                                    'token': len(P.text.split(" "))
                                },
                                'content': re.sub(whitespace_pattern, "\n\n", P.get_text(" ", strip=True))
                            }

                            # However, if the number in a numbered paragraph was missing, we will add the content to the previous paragraph.
                            if number_missing:
                                for paragraph in this_norm['paragraphs']:
                                    if str(paragraph['meta']['paragraph_id']) == str(number):
                                        paragraph['meta']['token'] += p_obj['meta']['token']
                                        paragraph['content'] += " " + p_obj['content']
                                        break

                            # Otherwise, we have a new paragraph.
                            else:
                                """
                                We will now do a final check if the paragraph we want to push might be a duplicate.
                                We will go through all paragraphs of the current norm and check.
                                For example, indmeterprobv has § 3 twice, which leads to a duplicate.
                                Original: https://www.gesetze-im-internet.de/indmeterprobv/__3.html
                                Duplicate: https://www.gesetze-im-internet.de/indmeterprobv/__3_1.html
                                """
                                hard_duplicate = False
                                for norm in output['norms']:
                                    if norm['meta']['norm_id'] == this_norm['meta']['norm_id']:
                                        for paragraph in norm['paragraphs']:
                                            if paragraph['meta']['paragraph_id'] == p_obj['meta']['paragraph_id']:
                                                # We found a duplicate
                                                hard_duplicate = True
                                                # sunprocessed_absatze.append(f"{filename} {key_process} {this_norm['meta']['norm_id']} {number}")
                                                break
                                # Only if we don't have a duplicate, we will push this paragraph.
                                if not hard_duplicate:
                                    this_norm['paragraphs'].append(p_obj)

                    # Pushing the fully processed norm to the output dict.
                    output['norms'].append(this_norm)
            
            # Selecting content from the output dict to return as text
            # Iterating over norms
            for key, value in output.items():
                text = ''
                for norm in value:
                    text += norm['meta']['title']
                    text += '\n'
                    for content in norm['paragraphs']:
                        text += content['content']
                        text += '\n'
            return text

    def get_texts(self):
        # Ensure the dataset is downloaded
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if len(os.listdir(self.output_dir)) == 0:
            self.download()

        filenames = [file for file in os.listdir(self.output_dir)]
        for filename in filenames:
            yield self.process_text(filename)