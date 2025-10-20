#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Helpers for reading from and writing to the SQLite backend.

The original project targeted Python 2.7, which relied on a couple of
language features that are no longer available in modern Python versions
(`basestring`, implicit binary/text conversions, etc.).  This module now
exposes the same API using only Python 3.12 compatible primitives while
keeping the runtime behaviour unchanged.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile

from .rstweb_reader import *  # noqa: F401,F403  (re-exported API)


with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
    DBPATH = tmpfile.name


def setup_db(dbpath=DBPATH):
    conn = sqlite3.connect(dbpath)

    cur = conn.cursor()

    # Drop tables if they exist
    cur.execute("DROP TABLE IF EXISTS rst_nodes")
    cur.execute("DROP TABLE IF EXISTS rst_relations")
    cur.execute("DROP TABLE IF EXISTS docs")
    cur.execute("DROP TABLE IF EXISTS perms")
    cur.execute("DROP TABLE IF EXISTS users")
    cur.execute("DROP TABLE IF EXISTS projects")
    cur.execute("DROP TABLE IF EXISTS logging")
    cur.execute("DROP TABLE IF EXISTS settings")
    conn.commit()

    # Create tables
    cur.execute('''CREATE TABLE IF NOT EXISTS rst_nodes
                 (id text, left real, right real, parent text, depth real, kind text, contents text, relname text, doc text, project text, user text, UNIQUE (id, doc, project, user) ON CONFLICT REPLACE)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS rst_relations
                 (relname text, reltype text, doc text, project text, UNIQUE (relname, reltype, doc, project) ON CONFLICT REPLACE)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS docs
                 (doc text, project text, user text,  UNIQUE (doc, project, user) ON CONFLICT REPLACE)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS users
                 (user text,  UNIQUE (user) ON CONFLICT REPLACE)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS projects
                 (project text, guideline_url text, UNIQUE (project) ON CONFLICT REPLACE)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS logging
                 (doc text, project text, user text, actions text, mode text, timestamp text)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS settings
                 (setting text, svalue text, UNIQUE (setting) ON CONFLICT REPLACE)''')

    conn.commit()
    conn.close()


def import_document(filename, project, user, dbpath=DBPATH):
    """parse and import an RS3 file into the local rstWeb SQLite database.

    Parameters
    ----------
    filename : unicode
        path to the RS3 file to be imported
    project : unicode
        name of an existing project that the file will be imported into
    user : str
        name of the user associated with that project
    """
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()

    doc=os.path.basename(filename)

    rel_hash = {}
    rst_nodes = read_rst(filename, rel_hash)
    if isinstance(rst_nodes, str):
        return rst_nodes

    # First delete any old copies of this document, if they are already imported
    delete_document(doc,project)

    for key in rst_nodes:
        node = rst_nodes[key]
        cur.execute("INSERT INTO rst_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (node.id,node.left,node.right,node.parent,node.depth,node.kind,node.text,node.relname,doc,project,user)) #user's instance
        cur.execute("INSERT INTO rst_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (node.id,node.left,node.right,node.parent,node.depth,node.kind,node.text,node.relname,doc,project,"_orig")) #backup instance

    for key in rel_hash:
        rel_name = key
        rel_type = rel_hash[key]
        cur.execute("INSERT INTO rst_relations VALUES(?,?,?,?)", (rel_name, rel_type, doc, project))

    cur.execute("INSERT INTO docs VALUES (?,?,?)", (doc,project,user))
    cur.execute("INSERT INTO docs VALUES (?,?,'_orig')", (doc,project))

    conn.commit()
    conn.close()


def get_rst_doc(doc, project, user, dbpath=DBPATH):
    """Return database representation of the given RS3 file.

    Parameters
    ----------
    doc : unicode
        filename of an already imported RS3 file
    project : unicode
        name of the project that the RS3 file was imported into
    user : unicode
        user name, e.g. 'admin' or 'local'

    Returns
    -------
    rows : list(tuple(unicode, float, float, unicode, float, unicode, unicode, unicode, unicode, unicode, unicode))
        A database representation of an RST tree / RS3 file.
        Each row contains these columns:
        - id (int ID of the EDU/segment/span, starting at 1, converted to unicode
        - left (int ID of leftmost EDU that this span covers, converted to float)
        - right (int ID of rightmost EDU that this span covers, converted to float)
        - parent (int ID of the parent segment/span of this one, converted to unicode)
        - depth ???, seems to be always 0.0
        - kind: 'edu', 'span' or 'multinuc'
        - contents: text of the EDU (if it is one, otherwise u'')
        - relname: RST relation name (w/ appended '_m' for multinuc and '_r' for 'rst')
        - doc: RS3 file name
        - project: name of the project the file belongs to
        - user: user name
    """
    conn = sqlite3.connect(dbpath)

    with conn:
        cur = conn.cursor()
        cur.execute("SELECT id, left, right, parent, depth, kind, contents, relname, doc, project, user FROM rst_nodes WHERE doc=? and project=? and user=? ORDER BY CAST(id AS int)", (doc,project,user))

        rows = cur.fetchall()
        return rows


def get_def_rel(relkind, doc, project):
    """Return the first RST relation name (e.g. 'conjunction') that is of the
    given relation type (e.g. 'multinuc').

    Parameters
    ----------
    relkind : str
        a type of RST relation, e.g. 'multinuc'
    doc : unicode
        filename of an already imported RS3 file
    project : unicode
        name of the project that the RS3 file was imported into

    Returns
    -------
    def_rel : unicode
        an RST relation name, e.g. 'conjunction'
    """
    rel_row = generic_query("SELECT relname FROM rst_relations WHERE reltype = ? and doc=? and project=? ORDER BY relname",(relkind,doc,project))
    return rel_row[0][0]


def get_rst_rels(doc, project, dbpath=DBPATH):
    """Return a list RST relations defined for the given document.

    The relations are not fetched directly from the file but from the
    local SQLite database into which the original file was parsed/imported.

    Note: RST relations have are appended with '_m' if they are of type
    'multinuc' and with '_r' in case of 'rst' relation types.

    Parameters
    ----------
    doc : unicode
        filename of an already imported RS3 file
    project : unicode
        name of the project that the RS3 file was imported into

    Returns
    -------
    rows : list(tuple(unicode, unicode))
        A list of (RST relation name, RST relation type) tuples,
        e.g. (u'restatement-mn_m', u'multinuc').
    """
    conn = sqlite3.connect(dbpath)

    with conn:
        cur = conn.cursor()
        cur.execute("SELECT relname, reltype FROM rst_relations WHERE doc=? and project=? ORDER BY relname", (doc,project))

        rows = cur.fetchall()
        return rows


def add_node(node_id,left,right,parent,rel_name,text,node_kind,doc,project,user):
    generic_query("INSERT INTO rst_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (node_id,left,right,parent,0,node_kind,text,rel_name,doc,project,user))


def update_parent(node_id,new_parent_id,doc,project,user):
    prev_parent = get_parent(node_id,doc,project,user)
    generic_query("UPDATE rst_nodes SET parent=? WHERE id=? and doc=? and project=? and user=?",(new_parent_id,node_id,doc,project,user))
    if new_parent_id == "0":
        update_rel(node_id,get_def_rel("rst",doc,project),doc,project,user)
    if new_parent_id != "0":
        if get_kind(new_parent_id,doc,project,user) =="multinuc":
            multi_rel = get_multirel(new_parent_id,node_id,doc,project,user)
            update_rel(node_id, multi_rel,doc,project,user)
        elif get_rel(node_id,doc,project,user)=="span" and get_kind(new_parent_id,doc,project,user) !="span": # A span child was just attached to a non-span
            update_rel(node_id, get_def_rel("rst",doc,project),doc,project,user)
    if prev_parent:
        if not count_children(prev_parent,doc,project,user)>0 and not prev_parent =="0": # Parent has no more children, delete it
            delete_node(prev_parent,doc,project,user)
        elif get_kind(prev_parent,doc,project,user)=="span" and count_span_children(prev_parent,doc,project,user)==0: # Span just lost its last span child, delete it
            delete_node(prev_parent,doc,project,user)
        elif get_kind(prev_parent,doc,project,user)=="multinuc" and count_multinuc_children(prev_parent,doc,project,user)==0: # Multinuc just lost its last multinuc child, delete it
            delete_node(prev_parent,doc,project,user)


# Returns the multinuclear relation with which a multinuc is currently dominating its children
def get_multirel(node_id,exclude_child,doc,project,user):
    rel_row = generic_query("SELECT rst_nodes.relname FROM rst_nodes JOIN rst_relations ON rst_nodes.relname = rst_relations.relname and rst_nodes.doc = rst_relations.doc and rst_nodes.project = rst_relations.project WHERE rst_relations.reltype = 'multinuc' and rst_nodes.parent=? and not rst_nodes.id=? and rst_nodes.doc=? and rst_nodes.project=? and rst_nodes.user=?",(node_id,exclude_child,doc,project,user))
    if len(rel_row) > 0:
        return rel_row[0][0]
    else:
        return get_def_rel("multinuc",doc,project)


def get_parent(node_id,doc,project,user):
    parent_row = generic_query("SELECT parent FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))
    return parent_row[0][0]


def get_rel(node_id,doc,project,user):
    rel_row = generic_query("SELECT relname FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))
    return rel_row[0][0]


def get_kind(node_id,doc,project,user):
    if node_id == "0":
        return "none"
    else:
        rel_row = generic_query("SELECT kind FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))
        return rel_row[0][0]


def update_rel(node_id,new_rel,doc,project,user):
    parent_id = get_parent(node_id,doc,project,user)
    if get_kind(parent_id,doc,project,user)=="multinuc":
        new_rel_type = get_rel_type(new_rel,doc,project)
        if new_rel_type == "rst":
            # Check if the last multinuc child of a multinuc just changed to rst
            if count_multinuc_children(parent_id,doc,project,user) == 1 and get_rel_type(get_rel(node_id,doc,project,user),doc,project) == "multinuc":
                new_rel = get_def_rel("rst",doc,project)
                children = get_children(parent_id,doc,project,user)
                for child in children:
                    update_parent(child[0],"0",doc,project,user)
            generic_query("UPDATE rst_nodes SET relname=? WHERE id=? and doc=? and project=? and user=?",(new_rel,node_id,doc,project,user))
        else: # New multinuc relation for a multinuc child, change all children to this relation
            generic_query("UPDATE rst_nodes SET relname=? WHERE id=? and doc=? and project=? and user=?",(new_rel,node_id,doc,project,user))
            children = get_children(parent_id,doc,project,user)
            for child in children:
                if get_rel_type(get_rel(child[0],doc,project,user),doc,project) == "multinuc":
                    generic_query("UPDATE rst_nodes SET relname=? WHERE id=? and doc=? and project=? and user=?",(new_rel,child[0],doc,project,user))
    else:
        generic_query("UPDATE rst_nodes SET relname=? WHERE id=? and doc=? and project=? and user=?",(new_rel,node_id,doc,project,user))


def count_children(node_id,doc,project,user):
    count = generic_query("SELECT count(*) FROM rst_nodes WHERE parent=? and doc=? and project=? and user=?",(node_id,doc,project,user))
    return int(count[0][0])


def count_multinuc_children(node_id,doc,project,user):
    count = generic_query("SELECT count(rst_nodes.id) FROM rst_nodes JOIN rst_relations ON rst_nodes.relname = rst_relations.relname and rst_nodes.doc = rst_relations.doc and rst_nodes.project = rst_relations.project WHERE reltype = 'multinuc' and parent=? and rst_nodes.doc=? and rst_nodes.project=? and user=?",(node_id,doc,project,user))
    return int(count[0][0])


def get_multinuc_children_lr(node_id,doc,project,user):
    lr = generic_query("SELECT min(rst_nodes.left), max(rst_nodes.right) FROM rst_nodes JOIN rst_relations ON rst_nodes.relname = rst_relations.relname and rst_nodes.doc = rst_relations.doc and rst_nodes.project = rst_relations.project WHERE reltype = 'multinuc' and parent=? and rst_nodes.doc=? and rst_nodes.project=? and user=?",(node_id,doc,project,user))
    return [int(lr[0][0]),int(lr[0][1])]


def get_multinuc_children_lr_ids(node_id,left,right,doc,project,user):
    id_left = generic_query("SELECT id FROM rst_nodes JOIN rst_relations ON rst_nodes.relname = rst_relations.relname and rst_nodes.doc = rst_relations.doc and rst_nodes.project = rst_relations.project WHERE reltype = 'multinuc' and parent=? and rst_nodes.left=? and rst_nodes.doc=? and rst_nodes.project=? and user=? ORDER BY rst_nodes.left",(node_id,left,doc,project,user))
    id_right = generic_query("SELECT id FROM rst_nodes JOIN rst_relations ON rst_nodes.relname = rst_relations.relname and rst_nodes.doc = rst_relations.doc and rst_nodes.project = rst_relations.project WHERE reltype = 'multinuc' and parent=? and rst_nodes.right=? and rst_nodes.doc=? and rst_nodes.project=? and user=? ORDER BY rst_nodes.left",(node_id,right,doc,project,user))
    return id_left[0][0],id_right[0][0]


def count_span_children(node_id,doc,project,user):
    count = generic_query("SELECT count(id) FROM rst_nodes WHERE relname = 'span' and parent=? and rst_nodes.doc=? and rst_nodes.project=? and user=?",(node_id,doc,project,user))
    return int(count[0][0])


def node_exists(node_id,doc,project,user):
    count = generic_query("SELECT count(*) FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))
    return int(count[0][0])>0


def get_rel_type(relname,doc,project):
    if relname=="span" or relname=="":
        return "span"
    else:
        return generic_query("SELECT reltype from rst_relations WHERE relname=? and doc=? and project=?",(relname,doc,project))[0][0]


def delete_node(node_id,doc,project,user):
    if node_exists(node_id,doc,project,user):
        parent = get_parent(node_id,doc,project,user)
        if not get_kind(node_id,doc,project,user) == "edu": # If it's not an EDU, it may be deleted
            # If there are still any children, such as rst relations to a deleted span or multinuc, set their parent to 0
            old_children = get_children(node_id,doc,project,user)
            for child in old_children:
                if len(child[0])>0:
                    update_parent(child[0],"0",doc,project,user)
            generic_query("DELETE FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))
        if not parent=="0":
            if not count_children(parent,doc,project,user)>0:
                delete_node(parent,doc,project,user)
            elif get_kind(parent,doc,project,user)=="span" and count_span_children(parent,doc,project,user)==0: # Span just lost its last span child, delete it
                delete_node(parent,doc,project,user)
            elif get_kind(parent,doc,project,user)=="multinuc" and count_multinuc_children(parent,doc,project,user)==0: # Multinuc just lost its last multinuc child, delete it
                delete_node(parent,doc,project,user)


def insert_parent(node_id,new_rel,node_kind,doc,project,user):
    lr = get_node_lr(node_id,doc,project,user)
    old_parent = get_parent(node_id,doc,project,user)
    old_rel = get_rel(node_id,doc,project,user)
    new_parent = str(get_max_node_id(doc,project,user) + 1)
    add_node(new_parent,lr[0],lr[1],old_parent,old_rel,"",node_kind,doc,project,user)
    update_parent(node_id,new_parent,doc,project,user)
    update_rel(node_id,new_rel,doc,project,user)


def get_children(parent,doc,project,user):
    return generic_query("SELECT id from rst_nodes WHERE parent=? and doc=? and project=? and user=?",(parent,doc,project,user))


def get_max_node_id(doc,project,user):
    return generic_query("SELECT max(CAST (id as decimal)) as max_id from rst_nodes WHERE doc=? and project=? and user=?",(doc,project,user))[0][0]


def get_max_right(doc,project,user):
    return generic_query("SELECT max(right) as max_right from rst_nodes WHERE doc=? and project=? and user=?",(doc,project,user))[0][0]


def generic_query(sql, params, dbpath=DBPATH):
    conn = sqlite3.connect(dbpath)

    with conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows


def delete_document(doc,project):
    generic_query("DELETE FROM rst_nodes WHERE doc=? and project=?",(doc,project))
    generic_query("DELETE FROM rst_relations WHERE doc=? and project=?",(doc,project))
    generic_query("DELETE FROM docs WHERE doc=? and project=?",(doc,project))


def insert_seg(token_num, doc, project, user):
    tok_seg_map = get_tok_map(doc,project,user)
    seg_to_split = tok_seg_map[token_num]
    push_up(int(seg_to_split),doc,project,user)
    parts = get_split_text(token_num,doc,project,user)
    update_seg_contents(seg_to_split,parts[0].strip(),doc,project,user)
    add_seg(str(int(seg_to_split)+1),parts[1].strip(),doc,project,user)


def get_tok_map(doc,project,user):
    rows = generic_query("SELECT id, contents FROM rst_nodes WHERE kind='edu' and doc=? and project=? and user=? ORDER BY CAST(id AS int)",(doc,project,user))
    all_tokens = {}
    token_counter = 0
    for row in rows:
        edu_text = row[1].strip()
        edu_tokens = edu_text.split(" ")
        for token in edu_tokens:
            token_counter += 1
            all_tokens[token_counter] = row[0]

    return all_tokens


def push_up(push_above_this_seg,doc,project,user):
    ids_above_push = generic_query("SELECT id from rst_nodes WHERE CAST(id as int) > ? and doc=? and project=? and user=? ORDER BY CAST(id as int) DESC",(push_above_this_seg,doc,project,user))
    for row in ids_above_push: #Do this row-wise to avoid sqlite unique constraint behavior
        id_to_increment = row[0]
        generic_query("UPDATE rst_nodes set id = CAST((CAST(id as int) + 1) as text) WHERE id=? and doc=? and project=? and user=?",(id_to_increment,doc,project,user))
    generic_query("UPDATE rst_nodes set parent = CAST((CAST(parent as int) + 1) as text) WHERE CAST(parent as int)>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))
    generic_query("UPDATE rst_nodes set left = left + 1 WHERE left>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))
    generic_query("UPDATE rst_nodes set right = right + 1 WHERE right>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))


def push_down(push_above_this_seg,doc,project,user):
    ids_above_push = generic_query("SELECT id from rst_nodes WHERE CAST(id as int) > ? and doc=? and project=? and user=? ORDER BY CAST(id as int)",(push_above_this_seg,doc,project,user))
    for row in ids_above_push: #Do this row-wise to avoid sqlite unique constraint behavior
        id_to_decrement = row[0]
        generic_query("UPDATE rst_nodes set id = CAST((CAST(id as int) - 1) as text) WHERE id=? and doc=? and project=? and user=?",(id_to_decrement,doc,project,user))
    generic_query("UPDATE rst_nodes set parent = CAST((CAST(parent as int) - 1) as text) WHERE CAST(parent as int)>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))
    generic_query("UPDATE rst_nodes set left = left - 1 WHERE left>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))
    generic_query("UPDATE rst_nodes set right = right - 1 WHERE right>? and doc=? and project=? and user=?",(push_above_this_seg,doc,project,user))


def get_split_text(tok_num,doc,project,user):
    rows = generic_query("SELECT id, contents FROM rst_nodes WHERE kind='edu' and doc=? and project=? and user=? ORDER BY CAST(id AS int)",(doc,project,user))
    token_counter = 0
    do_return = False
    final = []
    part1 = ""
    part2 = ""
    for row in rows:
        if do_return:
            do_return = False
            final = [part1.strip(),part2.strip()]
        else:
            part1 = ""
            part2 = ""
        edu_text = row[1].strip()
        edu_tokens = edu_text.split(" ")
        for token in edu_tokens:
            token_counter += 1
            if do_return == False:
                part1+=token + " "
            else:
                part2+=token + " "
            if tok_num == token_counter:
                do_return = True
    if do_return:
        final = [part1.strip(),part2.strip()]

    return final


def update_seg_contents(id,contents,doc,project,user):
    generic_query("UPDATE rst_nodes set contents=? WHERE id=? and doc=? and project=? and user=?",(contents,id,doc,project,user))


def get_seg_contents(id,doc,project,user):
    return generic_query("SELECT contents from rst_nodes WHERE id=? and doc=? and project=? and user=?",(id,doc,project,user))[0][0]


def add_seg(id,contents,doc,project,user):
    generic_query("INSERT INTO rst_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?)", (id,id,id,"0","0","edu",contents,get_def_rel("rst",doc,project),doc,project,user))

def merge_seg_forward(last_tok_num,doc,project,user):
    tok_seg_map = get_tok_map(doc,project,user)
    seg_to_merge_forward = tok_seg_map[last_tok_num]
    part1 = get_seg_contents(str(seg_to_merge_forward),doc,project,user)
    part2 = get_seg_contents(str(int(seg_to_merge_forward)+1),doc,project,user)
    update_seg_contents(seg_to_merge_forward,part1+" "+part2,doc,project,user)

    ### TODO: WE need to unlink not just the EDU, but also anybody who has a parent or ancestor of that should have parent=0

    #delete_node(get_parent(str(int(seg_to_merge_forward)+1),doc,project,user),doc,project,user)
    #unlink_children(str(int(seg_to_merge_forward)+1),doc,project,user)
    #unlink the edu marked for deletion
    update_parent(str(int(seg_to_merge_forward)+1),"0",doc,project,user)
    children = get_children(str(int(seg_to_merge_forward)+1),doc,project,user)
    #unlink its children
    for child in children:
        update_parent(child[0],"0",doc,project,user)
    #remove it from the database
    generic_query("DELETE FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(str(int(seg_to_merge_forward)+1),doc,project,user))
    push_down(int(seg_to_merge_forward),doc,project,user)


def get_node_lr(node_id,doc,project,user):
    left =  generic_query("SELECT left FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))[0][0]
    right =  generic_query("SELECT right FROM rst_nodes WHERE id=? and doc=? and project=? and user=?",(node_id,doc,project,user))[0][0]
    return [left,right]
