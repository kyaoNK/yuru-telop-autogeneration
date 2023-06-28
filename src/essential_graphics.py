from common import CONST_SPEAKER

from lxml.etree import parse
from lxml.etree import tostring
from lxml.etree import Element

import html

class EssentialGraphics():
    def __init__(self, text: str, starttime: int, endtime: int) -> Element:
        default_telop_filepath = 'default_telop.xml'
        self.clipitem_tag = parse(default_telop_filepath).getroot()

        start_tag = self.clipitem_tag.find('start')
        start_tag.text = str(starttime)

        end_tag = self.clipitem_tag.find('end')
        end_tag.text = str(endtime)

        text_tag = self.clipitem_tag.findall('filter/effect/name')[1]
        text_tag.text = text
    
    def set_id(self, clipitem_id:int, masterclip_id: int, file_id: int) -> None:
        self.clipitem_tag.attrib['id'] = 'clipitem-'+ str(clipitem_id)

        master_tag = self.clipitem_tag.find('masterclipid')
        master_tag.text = 'masterclip-' + str(masterclip_id)

        file_id_tag = self.clipitem_tag.find('file')
        file_id_tag.attrib['id'] = 'file-' + str(file_id)

eg = EssentialGraphics('こんにちは', 100000, 102202)