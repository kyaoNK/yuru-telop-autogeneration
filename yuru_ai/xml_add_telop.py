from lxml.etree import Element
from lxml.etree import SubElement
from lxml.etree import ElementTree

from lxml.etree import tostring
from lxml.etree import parse

YURU_AI_PATH = '/home/nlp-lab/kojima/yuru_telop_autogeneration/yuru_ai/'
src_filepath = YURU_AI_PATH + '2023_01_22 (8)_食レポ5_only.xml'
dest_filepath = YURU_AI_PATH + '2023_01_22 (8)_食レポ5_add_telop.xml'
csv_filepath = 

def parse_xml(filepath: str) -> Element :
    tree = parse(filepath)
    root = tree.getroot()
    return root

def write_xml(root: Element , filepath: str) :
    tree = ElementTree(root)
    with open(filepath, 'wb') as file :
        tree.write(
            file,
            encoding='utf-8',
            xml_declaration=True,
            pretty_print=True,
            doctype='<!DOCTYPE xmeml>'
        )

class telop():
    def __init__(self, text: str) -> None:
        self.text = text

class serif():
    def __init__(self, text: str) -> None:
        self.text = text


def add_telop_serif(root: Element) -> Element :




    return root

root = parse_xml(src_filepath)
root = add_telop(root, csv_filepath)
write_xml(root, dest_filepath)
