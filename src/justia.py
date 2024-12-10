import json
import time
from selenium import webdriver
from bs4 import BeautifulSoup
from typing import List
from enum import Enum
from dataclasses import dataclass, asdict, fields
import argparse
from datetime import datetime
import requests
from tqdm import tqdm
import os
import csv
import re
import html
import unicodedata


home_url = "https://supreme.justia.com"
by_year_url = "https://supreme.justia.com/cases/federal/us/year"
href_pref = "/cases/federal/us/"


@dataclass
class Case:
    docket: str
    citation: str
    year: int
    url: str

@dataclass
class Opinion:
    case: Case
    id: int
    author: str
    type: str
    text: str
    joined: List[str]



with open("/Users/danil_sibgatullin/6.8611/per-curiam/json/justices.json", "r") as file:
    justices = json.load(file)
serving_justices = {justice['name'] for justice in justices}

def get_cases_url(year: int) -> List[Case]:
    url = f"{by_year_url}/{year}.html"

    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(1) 
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    cases, dockets = [], set()

    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith(href_pref) and len(href) > len(href_pref):
            docket = href.split('/')[-2]
            citation =  href.split('/')[-3]
            case_link = f"{home_url.rstrip('/')}/{href.lstrip('/')}"
            if not docket in dockets:
                dockets.add(docket)
                cases.append(Case(docket=docket, citation=citation, url=case_link, year=year))

    driver.quit()

    return cases

def clean_text(text: str) -> str:
    # Normalize Unicode to ASCII and remove non-ASCII characters
    # return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text

def parse_case_opinions(case: Case) -> List[Opinion]:
    response = requests.get(case.url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    opinions = []   

    opinion_sections = soup.find_all('a', id=lambda x: x and x.startswith('list-opinion-'))

    for section in opinion_sections:
        element_text = section.get_text(strip=True)
        author = element_text.split('(')[-1].replace(')', '').strip().lower()

        if author != 'per curiam':
            continue

        tab = soup.find_all('div', id=section['href'][1:])[0]
        
        
        opinions.append(Opinion(case=case, 
                                author=author, 
                                type=section.get_text(strip=True).split('(')[0].strip().lower(),
                                id=section['href'].split('-')[-1],
                                text= clean_text(tab.get_text(strip=True)),
                                joined=[])
        )

    return opinions

def dump_cases_url(year_from: int, year_to: int, filename: str):
    all_cases = sum([get_cases_url(y) for y in range(year_from, year_to + 1)], [])
    all_cases = [asdict(case) for case in all_cases]
    with open(filename, "w") as json_file:
        json.dump(all_cases, json_file, indent=4)

def dump_case_texts(cases_file: str, opinions_dir: str, summary_csv: str):
    # loading the lists of cases
    with open(cases_file, "r") as file:
        cases = json.load(file)
    cases = [Case(**case) for case in cases]
    cases = [case for case in cases if case.year >= 2006]

    os.makedirs(opinions_dir, exist_ok=True)

    summary_data = []
    case_headers = ["Year", "Docket", "Citation", "URL"]
    for case in tqdm(cases):
        opinions = parse_case_opinions(case)

        # write each opinion
        for opinion in opinions:
            opinion_file = os.path.join(opinions_dir, f"{opinion.author}_{case.year}_{case.docket}_{case.citation}.json")
            with open(opinion_file, "w") as json_file:
                json.dump(asdict(opinion), json_file, indent=4)

        # write the summary table
        row = {field: getattr(case, field.lower()) for field in case_headers}
        for justice in justices:
            justice_opinions = [opinion.type for opinion in opinions if opinion.author == justice['name']]
            row[justice['name']] = ", ".join(justice_opinions) if justice_opinions else ""

        summary_data.append(row)

    # Write the summary CSV
    with open(summary_csv, "w", newline="") as csv_file:
        fieldnames = case_headers + [justice.name for justice in justices]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Supreme Court cases and opinions.")
    
    # Required cases file
    parser.add_argument("-c", "--cases", required=True, help="Path to the cases JSON file to process")
    
    # Optional opinions and summary
    parser.add_argument("-o", "--opinions", help="Path to the directory to save opinions")
    parser.add_argument("-s", "--summary", help="Path to the file to save the summary CSV")

    args = parser.parse_args()

    # Process opinions and summary
    if args.opinions and args.summary:
        # Ensure the opinions directory exists
        os.makedirs(args.opinions, exist_ok=True)
        print(f"Scraping texts from case links and saving in {args.opinions}")
        print(f"Generating summary CSV and saving at {args.summary}")
        dump_case_texts(args.cases, args.opinions, args.summary)
    else:
        justices_file = "/Users/danil_sibgatullin/6.8611/per-curiam/json/justices.json"
        with open(justices_file, "r") as file:
            justices = json.load(file)
        year_from = min([justice['year_joined'] for justice in justices])
        year_to = datetime.today().year

        # Process cases
        print(f"Scraping links from {year_from} to {year_to}")
        dump_cases_url(year_from, year_to, args.cases)

