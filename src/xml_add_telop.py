from telop import Telop
from clipid_manager import ClipIDManager
from common import CONST_SPEAKER
from util import datetime2nframe

from lxml.etree import Element
from lxml.etree import tostring
from lxml.etree import fromstring
from lxml.etree import parse
from lxml.etree import indent
from lxml.etree import XML

import xml.dom.minidom

import srt
import os
import re
import html

XML_PATH = 'xml/'
CSV_PATH = 'csv/'
TXT_PATH = 'txt/'

def parse_srt(txt_filepath :str) -> list:
    with open(txt_filepath, 'r') as file:
        srt_data = list(srt.parse(file.read()))
        return srt_data

def parse_xml(filepath: str) -> Element :
    # print(filepath)
    tree = parse(filepath)
    root = tree.getroot()
    return root

def add_telop_serif(root: Element, srt_data: list) -> Element :
    video_tag = root.find('sequence/media/video')
    track_tags = video_tag.findall('track')
    speaker1_telop_track_tag = track_tags[1]
    # speaker2_telop_track_tag = track_tags[2]
    # speaker1_serif_track_tag = track_tags[3]
    # speaker2_serif_track_tag = track_tags[4]
    
    # append masterclipid and clipitemid to clipid_manager
    clipid_manager = ClipIDManager(root)
    print('========== before ==========')
    clipid_manager.print()

    for d in srt_data:
        # print(f'{d.start} -> {d.end} : {datetime2nframe(d.start)} -> {datetime2nframe(d.end)}\n{d.content}')
        new_clipitemid = clipid_manager.get_new_clipitemid()
        new_masterclipid = clipid_manager.get_new_masterclipid()
        new_fileid = clipid_manager.get_new_fileid()
        
        new_telop = Telop(d.content, CONST_SPEAKER['horimoto'], datetime2nframe(d.start), datetime2nframe(d.end))
        new_telop.essential_graphics.set_id(clipitem_id=new_clipitemid, masterclip_id=new_masterclipid, file_id=new_fileid)
        speaker1_telop_track_tag.insert(-1, new_telop.essential_graphics.clipitem_tag)

    print('========== after ==========')
    clipid_manager.print()
    return root

src_filepath = XML_PATH + '2023_01_22 (8)_食レポ5_60_only.xml'
srt_filepath = TXT_PATH + '2023_01_22 (8)_食レポ5_編集なし_ショート_whipserX_segments.srt'
dest_filepath = XML_PATH + '2023_01_22 (8)_食レポ5_add_telop.xml'

root = parse_xml(src_filepath)
srt_data = parse_srt(srt_filepath)
root = add_telop_serif(root, srt_data)

# root_str = tostring(root, pretty_print=True).decode('utf-8')
root_str = tostring(root, encoding='utf-8').decode('utf-8')

root_str = re.sub(r'[\t\n]+', '', root_str)
# print(root_str[0:40])
root = XML(root_str)
indent(root, space='\t')
root_str = tostring(root, encoding='utf-8').decode('utf-8')

with open(dest_filepath, 'w') as f:
    f.write(root_str)


