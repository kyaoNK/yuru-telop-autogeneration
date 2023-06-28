from lxml.etree import Element
from lxml.etree import SubElement
from lxml.etree import ElementTree

from lxml.etree import tostring
from lxml.etree import parse

import html

def parse_xml(filepath: str) -> Element :
    # print(filepath)
    tree = parse(filepath)
    root = tree.getroot()
    return root

src_filepath = 'xml/2023_01_22 (8)_食レポ5_60_完成版.xml'
root = parse_xml(src_filepath)
video_tag = root.find('sequence/media/video')
track_tags = video_tag.findall('track')
speaker1_telop_track_tag = track_tags[1]
clipitem_tags = speaker1_telop_track_tag.findall('clipitem')
defalut_telop = clipitem_tags[1]
defalut_telop_str = tostring(defalut_telop, pretty_print=True).decode('utf-8')
defalut_telop_str = html.unescape(defalut_telop_str)
defalut_telop_str = '\t\t\t\t\t' + defalut_telop_str
# print(tostring(defalut_telop, pretty_print=True).decode('utf-8'))
with open('default_telop_60.xml', 'w') as f:
    f.write(defalut_telop_str)