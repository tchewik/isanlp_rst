#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Utilities for parsing ``.rs3`` and text files into in-memory objects."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from .rstweb_classes import NODE, get_left_right


def read_rst(filename, rel_hash):
	"""parse an RS3 file into a representation that can be stored in SQLite.

	Note: `rel_hash` is never returned, so the calling function defines it
	as an empty dict and passes it to `read_rst`. After the function
	is run, the calling function has a filled `rel_hash`, cf.
	https://pythonconquerstheuniverse.wordpress.com/category/python-gotchas/

	Parameters
	----------
	filename : unicode
		path to the RS3 file to be imported
	rel_hash : dict
		dict into which the RST relation names / types defined in the file
		are stored (cf. `rstweb_sql.get_rst_rels()`)

	Returns
	-------
	elements : dict(str: NODE)
		a map from a node ID to a NODE instance. Each NODE represents an
		element from an RST tree (and links to its parent NODE)
	"""
	try:
		with open(filename, "r", encoding="utf-8") as f:
			xml_content = f.read()
	except OSError as err:
		return f"Unable to read '{filename}': {err.strerror}."

	try:
		xmldoc = minidom.parseString(xml_content)
	except ExpatError:
		message = "Invalid .rs3 file"
		return message

	nodes = []
	ordered_id = {}
	schemas = []
	default_rst = ""

	# Get relation names and their types, append type suffix to disambiguate
	# relation names that can be both RST and multinuc
	item_list = xmldoc.getElementsByTagName("rel")
	for rel in item_list:
		relname = re.sub(r"[:;,]","",rel.attributes["name"].value)
		if rel.hasAttribute("type"):
			rel_hash[relname+"_"+rel.attributes["type"].value[0:1]] = rel.attributes["type"].value
			if rel.attributes["type"].value == "rst" and default_rst=="":
				default_rst = relname+"_"+rel.attributes["type"].value[0:1]
		else:  # This is a schema relation
			schemas.append(relname)


	item_list = xmldoc.getElementsByTagName("segment")
	if len(item_list) < 1:
		return '<div class="warn">No segment elements found in .rs3 file</div>'

	id_counter = 0


	# Get hash to reorder EDUs and spans according to the order of appearance in .rs3 file
	for segment in item_list:
		id_counter += 1
		ordered_id[segment.attributes["id"].value] = id_counter
	item_list = xmldoc.getElementsByTagName("group")
	for group in item_list:
		id_counter += 1
		ordered_id[group.attributes["id"].value] = id_counter
	ordered_id["0"] = 0

	element_types={}
	node_elements = xmldoc.getElementsByTagName("segment")
	for element in node_elements:
		element_types[element.attributes["id"].value] = "edu"
	node_elements = xmldoc.getElementsByTagName("group")
	for element in node_elements:
		element_types[element.attributes["id"].value] = element.attributes["type"].value

	id_counter = 0
	item_list = xmldoc.getElementsByTagName("segment")
	for segment in item_list:
		id_counter += 1
		if segment.hasAttribute("parent"):
			parent = segment.attributes["parent"].value
		else:
			parent = "0"
		if segment.hasAttribute("relname"):
			relname = segment.attributes["relname"].value
		else:
			relname = default_rst

		# Tolerate schemas, but no real support yet:
		if relname in schemas:
			relname = "span"

			relname = re.sub(r"[:;,]","",relname) #remove characters used for undo logging, not allowed in rel names
		# Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
		if parent in element_types:
			if element_types[parent] == "multinuc" and relname+"_m" in rel_hash:
				relname = relname+"_m"
			elif relname !="span":
				relname = relname+"_r"
		else:
			if not relname.endswith("_r") and len(relname)>0:
				relname = relname+"_r"
		edu_id = segment.attributes["id"].value
		contents = segment.childNodes[0].data.strip()
		nodes.append([str(ordered_id[edu_id]), id_counter, id_counter, str(ordered_id[parent]), 0, "edu", contents, relname])

	item_list = xmldoc.getElementsByTagName("group")
	for group in item_list:
		if group.attributes.length == 4:
			parent = group.attributes["parent"].value
		else:
			parent = "0"
		if group.attributes.length == 4:
			relname = group.attributes["relname"].value
			# Tolerate schemas by treating as spans
			if relname in schemas:
				relname = "span"
				
			relname = re.sub(r"[:;,]","",relname) #remove characters used for undo logging, not allowed in rel names
			# Note that in RSTTool, a multinuc child with a multinuc compatible relation is always interpreted as multinuc
			if parent in element_types:
				if element_types[parent] == "multinuc" and relname+"_m" in rel_hash:
					relname = relname+"_m"
				elif relname !="span":
					relname = relname+"_r"
			else:
				relname = ""
		else:
			relname = ""
		group_id = group.attributes["id"].value
		group_type = group.attributes["type"].value
		contents = ""
		nodes.append([str(ordered_id[group_id]),0,0,str(ordered_id[parent]),0,group_type,contents,relname])


	elements = {}
	for row in nodes:
		elements[row[0]] = NODE(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],"")

	for element in elements:
		if elements[element].kind == "edu":
			get_left_right(element, elements,0,0,rel_hash)

	return elements


def read_text(filename,rel_hash):
	id_counter = 0
	nodes = {}
	with open(filename, "r", encoding="utf-8") as f:
		lines = f.readlines()
	#Add some default relations if none have been supplied (at least 1 rst and 1 multinuc)
	if len(rel_hash) < 2:
		rel_hash["elaboration_r"] = "rst"
		rel_hash["joint_m"] = "multinuc"

	rels = OrderedDict(sorted(rel_hash.items()))

	try:
		first_relname, first_reltype = next(iter(rels.items()))
	except StopIteration:
		raise ValueError("Relation map is empty; expected at least one relation.")

	for line in lines:
		id_counter += 1
		nodes[str(id_counter)] = NODE(
			str(id_counter),
			id_counter,
			id_counter,
			"0",
			0,
			"edu",
			line.strip(),
			first_relname,
			first_reltype,
		)

	return nodes


def read_relfile(filename):
	with open(filename, "r", encoding="utf-8") as f:
		rel_lines = f.readlines()

	rels: Dict[str, str] = {}
	for line in rel_lines:
		if line.find("\t") > 0:
			rel_data = line.split("\t")
			if rel_data[1].strip() == "rst":
				rels[rel_data[0].strip()+"_r"]="rst"
			elif rel_data[1].strip() == "multinuc":
				rels[rel_data[0].strip()+"_m"]="multinuc"

	return rels
