import xml.etree.ElementTree as ET
import pytest
from pathlib import Path
from usv.auto_annotation.validation.xml_schema_test import validate_xml


def _write_xml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "annotations.xml"
    p.write_text(content, encoding="utf-8")
    return p


VALID_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="1" label="Vessel" z_order="30" source="auto">
    <polygon frame="0" points="0,0;10,0;10,10;0,10"
             outside="0" occluded="0" keyframe="1"/>
    <polygon frame="5" points="1,1;11,1;11,11;1,11"
             outside="0" occluded="0" keyframe="1"/>
  </track>
</annotations>"""


def test_valid_xml_passes(tmp_path):
    xml_path = _write_xml(tmp_path, VALID_XML)
    violations = validate_xml(xml_path)
    assert violations == []


def test_wrong_root_tag(tmp_path):
    xml = VALID_XML.replace("<annotations>", "<annotation>").replace("</annotations>", "</annotation>")
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("Root tag" in v for v in violations)


def test_missing_track_label(tmp_path):
    xml = VALID_XML.replace('label="Vessel" ', '')
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("label" in v for v in violations)


def test_missing_track_z_order(tmp_path):
    xml = VALID_XML.replace(' z_order="30"', '')
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("z_order" in v for v in violations)


def test_duplicate_track_ids(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="1" label="Vessel" z_order="30">
    <polygon frame="0" points="0,0;10,0;10,10" outside="0" occluded="0" keyframe="1"/>
  </track>
  <track id="1" label="Buoy" z_order="50">
    <polygon frame="0" points="0,0;10,0;10,10" outside="0" occluded="0" keyframe="1"/>
  </track>
</annotations>"""
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("Duplicate" in v for v in violations)


def test_track_with_no_polygons(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="1" label="Vessel" z_order="30">
  </track>
</annotations>"""
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("no <polygon>" in v for v in violations)


def test_all_polygons_outside(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="1" label="Vessel" z_order="30">
    <polygon frame="0" points="0,0;10,0;10,10" outside="1" occluded="0" keyframe="1"/>
    <polygon frame="1" points="0,0;10,0;10,10" outside="1" occluded="0" keyframe="1"/>
  </track>
</annotations>"""
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("all polygons" in v for v in violations)


def test_no_keyframe_polygon(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <track id="1" label="Vessel" z_order="30">
    <polygon frame="0" points="0,0;10,0;10,10" outside="0" occluded="0" keyframe="0"/>
  </track>
</annotations>"""
    xml_path = _write_xml(tmp_path, xml)
    violations = validate_xml(xml_path)
    assert any("no polygon with keyframe" in v for v in violations)


def test_malformed_xml(tmp_path):
    xml_path = _write_xml(tmp_path, "<annotations><track id=1></annotations>")
    violations = validate_xml(xml_path)
    assert any("parse error" in v.lower() for v in violations)
