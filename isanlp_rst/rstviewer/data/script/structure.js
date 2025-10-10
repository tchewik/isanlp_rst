/**
 * Client side script for editing rhetorical relations.
 * The initial layout is calculated by the corresponding Python
 * script and saving updates the database and re-renders. Links
 * between nodes are made with jsPlumb and updated via the function
 * recalculate_depth(nodes) which is fed from the data model stored
 * in the hidden input "data".
 * Author: Amir Zeldes
*/


//TODO: get_max_node_id(nodes) can be implemented without reference to nodes, by parsing hidden data model

function act(action){

    disable_buttons();

    if (document.getElementById("undo_state").value != "undo"){//do not log undo actions for save in undo_state: let them get popped
        if (document.getElementById("action").value == ''){
            document.getElementById("action").value = action;
            document.getElementById("dirty").value = "dirty";}
        else{
            document.getElementById("action").value += ";" + action;
        }
        if (document.getElementById("undo_state").value == ""){
            document.getElementById("redo_log").value =""; //new non-undo, non-redo action, reset redo_log
        }
    }

    var action_type = action.split(":")[0];
    var action_params = action.split(":")[1];
    var params = action_params.split(",");
    nodes = parse_data();

    document.getElementById("logging").value += action;
    if (action_type == 'mn' || action_type == 'sp'){
        new_id = get_max_node_id(nodes) + 1;
        document.getElementById("logging").value += "," + new_id
    }
    if (document.getElementById("undo_state").value == ""){
        log_undo = "normal";
    }
    else{
        log_undo = document.getElementById("undo_state").value;
    }
    document.getElementById("logging").value += "," + log_undo + ";";

    def_multirel = get_def_multirel();
    if (action_type =="up" || action_type =="qup"){
        if (params[1]!="0"){
            if (nodes["n"+params[0]].parent != "n0"){
                if (nodes[nodes["n"+params[0]].parent].kind == "multinuc"){ //if previous parent was a multinuc, also note relation
                    append_undo("qrl:" + params[0] + "," + nodes["n"+params[0]].relname+"+");
                }
            }
        }
		append_undo("up:" + params[0] + "," + nodes["n"+params[0]].parent.replace("n",""));
        update_parent("n"+params[0],"n"+params[1]);
        nodes = parse_data();
    }
    else if (action_type =="sp"){
        append_undo("xx:" +  (get_max_node_id(nodes) + 1) + "," + params[0]);
        insert_parent("n"+params[0],"span","span");
    }
    else if (action_type =="mn"){
        append_undo("xx:" +  (get_max_node_id(nodes) + 1) + "," + params[0]);
        insert_parent("n"+params[0],def_multirel,"multinuc");
    }
    else if (action_type =="rl" || action_type=="qrl"){ //explicit relation change or quiet relation change in undo
        append_undo("rl:" + params[0] + "," + nodes["n"+params[0]].relname);
        if (nodes["n"+params[0]].parent != "n0"){
            if (nodes[nodes["n"+params[0]].parent].kind=="multinuc" && nodes[nodes["n"+params[0]].parent].parent == "n0" && get_rel_type(params[1])=="rst" && count_multinuc_children(nodes["n"+params[0]].parent,nodes)<2 && get_rel_type(nodes["n"+params[0]].relname)=="multinuc"){
            //attempted to change last multinuc child of a root node to RST - abort
                alert("Unable to change last child of unlinked multinuc to satellite relation!");
                enable_buttons();
                return;
            }
        }
        update_rel("n"+params[0],params[1],nodes);
        recalculate_depth(parse_data());
    }
    else if (action_type=="qnd"){ //quiet node existence restored in undo
	    add_node("n"+params[0],"n0",params[2].substring(0,1),params[3],params[4]);
        create_node_div(params[0],0,params[1],params[1],"0.5")
    }
    else if (action_type=="xx"){ //kill span or multinuc as undo of sp: or mn:
        rel_to_parent = nodes["n"+params[0]].relname;
        new_parent_id = nodes["n"+params[0]].parent;

        //connect child of element to delete with parent of element to delete
        update_data("n"+params[1],"n"+params[1]+","+new_parent_id+","+nodes["n"+params[1]].kind.substring(0,1)+","+nodes["n"+params[1]].left.toString()+","+rel_to_parent+","+get_rel_type(rel_to_parent));

        remove_node_data("n"+params[0]);
        detach_source("g"+params[0]);
        detach_target("g"+params[0]);
        detach_source("lg"+params[0]);
        detach_target("lg"+params[0]);
        document.getElementById("lg"+params[0]).style.display = "none";
        document.getElementById("g"+params[0]).style.display = "none";
        recalculate_depth(parse_data());
    }

    // anim_catch replaces jquery promise to monitor animation queue
    // enable buttons once this final animation is complete
    if (document.getElementById("anim_catch").style.left=="10px"){
        jsPlumb.animate("anim_catch",{left: 20},complete=function(){enable_buttons();});
        }
    else{
        jsPlumb.animate("anim_catch",{left: 10},complete=function(){enable_buttons();});
        }
}

function crel(node_id,sel) {act("rl:" + node_id.toString() + "," + sel);}

function rst_node(id, parent, kind, left, relname, reltype){

    switch(kind) {
    case "s":
        this.kind = "span";
        this.left = 0;
        this.right = 0;
        break;
    case "m":
        this.kind = "multinuc";
        this.left = 0;
        this.right = 0;
        break;
    default:
        this.kind = "edu"
        this.left = parseInt(left);
        this.right = this.left;
        break;
    }

    this.id = id;
    this.parent = parent;
    this.relname = relname;
    this.reltype = reltype;
    this.depth = 0;
    this.anchor = 0;
    return this;

}

function parse_data() {
    var data_string = document.getElementById("data").value;
    var node_array = data_string.split(";");
    var i;
    var nodes = {};
    var node_args =[];

    for (i = 0; i < node_array.length; i++) {
        node_args = node_array[i].split(",");
        node_id = node_args[0];
        nodes[node_id] = {};
        nodes[node_id] = new rst_node(node_id,node_args[1],node_args[2],node_args[3],node_args[4],node_args[5]);
    }

    //calculate RST effective depth
    for (node_id in nodes){
        get_depth(node_id,node_id,nodes);
    }

    //calculate left and right span borders for each node
    for (node_id in nodes){
        if (nodes[node_id].kind =="edu"){
            get_left_right(node_id,nodes,0,0);
        }
    }
    anchors = get_anchor_points(nodes);
    for (node_id in nodes){
        nodes[node_id].anchor = anchors[node_id];
    }

    return nodes;
}

function update_parent(node_id,new_parent_id){
    var nodes = {};
    nodes = parse_data();
	prev_parent = nodes[node_id].parent;
    append_undo("+qrl:"+node_id.replace("n","")+","+nodes[node_id].relname);

	if (new_parent_id == "n0"){
		update_data(node_id,node_id+","+new_parent_id+","+nodes[node_id].kind.substring(0,1)+","+nodes[node_id].left.toString()+","+get_def_rstrel()+",rst");
	}
	else
	{
	    update_data(node_id,node_id+","+new_parent_id+","+nodes[node_id].kind.substring(0,1)+","+nodes[node_id].left.toString()+","+nodes[node_id].relname+","+nodes[node_id].reltype);
	}

    nodes = parse_data();

	if (new_parent_id != "n0"){
		if (nodes[new_parent_id].kind =="multinuc"){
			multi_rel = get_multirel(new_parent_id,node_id,nodes);
			update_rel(node_id, multi_rel,nodes);
			if ($("#sel" + node_id.replace("n","")).length == 0){ //if the select has been deleted, make sure it is rendered again
			       recalculate_depth(nodes);
			}
			document.getElementById('sel'+node_id.replace("n","")).value = multi_rel;

		}
	}
	if (prev_parent!="n0"){
		if (count_children(prev_parent,nodes)==0 && prev_parent !="n0"){// Parent has no more children, delete it
			delete_node(prev_parent,nodes);
		}
		else if (nodes[prev_parent].kind=="span" && count_span_children(prev_parent,nodes)==0){ // Span just lost its last span child, delete it
			delete_node(prev_parent,nodes);
		}
		else if (nodes[prev_parent].kind=="multinuc" && count_multinuc_children(prev_parent,nodes)==0){ // Multinuc just lost its last multinuc child, delete it
			delete_node(prev_parent,nodes);
		}
	}
    recalculate_depth(parse_data());

}

function update_data(node_id,new_data){
    var re = new RegExp("(^|;)" + node_id + ",[^;]+","g");
    document.getElementById("data").value = document.getElementById("data").value.replace(re,"$1" + new_data);
}


function remove_node_data(node_id){
    var re = new RegExp("(^|;)" + node_id + ",[^;]+","g");
    document.getElementById("data").value = document.getElementById("data").value.replace(re,"");
}


// This function returns the multinuclear relation with which a multinuc is currently dominating its children
function get_multirel(multinuc_id,exclude_child,nodes){
    found = false;
    for (node_id in nodes){
        if (node_id != exclude_child && nodes[node_id].parent == multinuc_id){
            found ==true;
            return nodes[node_id].relname;
        }
    }
    if (found == false){
       return get_def_multirel();
    }
}

// Get default multinuc relation from hidden field
function get_def_multirel(){
    return document.getElementById("def_multi_rel").value;
}

// Get default rst relation from hidden field
function get_def_rstrel(){
    return document.getElementById("def_rst_rel").value;
}

function update_rel(node_id,new_rel,nodes){
	parent_id = nodes[node_id].parent;
    if (parent_id == "n0"){
        parent_kind = "none";
    }
    else{
	    parent_kind = nodes[parent_id].kind;
	}
	new_rel_type = get_rel_type(new_rel);
	old_rel_type = nodes[node_id].reltype;
    parent_element_id = "g"+parent_id.replace("n","");
    if (nodes[node_id].kind == "edu"){
        element_id = "edu"+node_id.replace("n","");
    }
    else{
        element_id = "g"+node_id.replace("n","");
    }

	if (parent_kind=="multinuc" && parent_id!="n0"){

		if (new_rel_type == "rst"){
			// Check if the last multinuc child of a multinuc just changed to rst
			if (count_multinuc_children(parent_id,nodes) == 1 && nodes[node_id].reltype  == "multinuc"){
                new_rel = get_def_rstrel();
				children = get_children(parent_id,nodes);
                for (var i=0; i<children.length; i++){
					update_parent(children[i],"n0");
				}
            }
            else if(nodes[node_id].reltype  == "multinuc"){ //a non-last multinuc child was changed to rst
                detach_source(element_id);
                jsPlumb.connect({source: element_id,target:parent_element_id, overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }], ["Custom", {create:function(component) {return make_relchooser(node_id,"rst",new_rel);},location:0.2,id:"customOverlay"}]]});
            }
            update_data(node_id,node_id+","+nodes[node_id].parent+","+nodes[node_id].kind.substring(0,1)+","+nodes[node_id].left.toString()+","+new_rel+","+new_rel_type);
            if ($("#sel"+node_id.replace("n","")).length > 0){
                document.getElementById('sel'+node_id.replace("n","")).value = new_rel;
            }
		}
		else { // New multinuc relation for a multinuc child, change all children to this relation
            update_data(node_id,node_id+","+nodes[node_id].parent+","+nodes[node_id].kind.substring(0,1)+","+nodes[node_id].left.toString()+","+new_rel+","+new_rel_type);
            if (new_rel!="span") {
                if (old_rel_type=="rst"){//this is an rst satellite that has been restored to multinuc child via undo
                    detach_source(element_id);
                    jsPlumb.connect({source: element_id,target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(node_id,"multi",new_rel);},location:0.2,id:"customOverlay"}]]});
                }
                children = get_children(parent_id,nodes);
                for (var i=0; i<children.length; i++){
                    if (nodes[children[i]].reltype == "multinuc" && nodes[children[i]].id != node_id){
                        update_data(children[i],children[i]+","+nodes[children[i]].parent+","+nodes[children[i]].kind.substring(0,1)+","+nodes[children[i]].left.toString()+","+new_rel+","+new_rel_type);
                        if ($("#sel"+children[i].replace("n","")).length > 0){
                            document.getElementById('sel'+children[i].replace("n","")).value = new_rel;
                        }
                    }
                }
            }
		}
	}
	else{
        update_data(node_id,node_id+","+nodes[node_id].parent+","+nodes[node_id].kind.substring(0,1)+","+nodes[node_id].left.toString()+","+new_rel+","+new_rel_type);
	}
}

function get_rel_type(relname){
    if (relname =="span"){
        return "span";
    }
    else {
        if (relname.endsWith("_m")){
            return "multinuc";
        }
        else
        {
            return "rst";
        }
    }
}


function get_children(parent,nodes){
    ids = [];
    for (node_id in nodes){
        if (nodes[node_id].parent == parent) {ids.push(node_id);}
    }
    return ids;
}

function count_multinuc_children(parent,nodes) {
    ids = [];
    for (node_id in nodes){
        if (nodes[node_id].parent == parent && nodes[node_id].reltype =="multinuc") {ids.push(node_id);}
    }
    return ids.length;
}

function count_span_children(parent,nodes) {
    ids = [];
    for (node_id in nodes){
        if (nodes[node_id].parent == parent && nodes[node_id].relname =="span") {ids.push(node_id);}
    }
    return ids.length;
}

function count_children(parent,nodes) {
    ids = [];
    for (potential_child in nodes){
        if (nodes[potential_child].parent == parent) {ids.push(potential_child);}
    }
    return ids.length;
}

function delete_node(node_id,nodes){

    if (nodes[node_id].kind != "edu"){
        element_id = "g"+node_id.replace("n","");
    }
    else
    {
        element_id = "edu"+node_id.replace("n","");
    }
    if (node_exists(node_id) && document.getElementById(element_id).style.display!="none"){
        former_parent = nodes[node_id].parent;
        if (nodes[node_id].kind != "edu"){ //If it's not an EDU, it may be deleted

            document.getElementById("lg"+node_id.replace("n","")).style.display = "none";
            document.getElementById("g"+node_id.replace("n","")).style.display = "none";

            // If there are still any children, such as rst relations to a deleted span or multinuc, set their parent to n0
            old_children = get_children(node_id,nodes);
            for (var i=0; i<old_children.length; i++){
                old_child_id=old_children[i];
                update_parent(old_children[i],"n0");
                append_undo("+qup:"+old_child_id.replace("n","")+","+node_id.replace("n",""));
            }

            append_undo("+qup:"+node_id.replace("n","")+","+former_parent.replace("n",""));
            append_undo("+qnd:"+node_id.replace("n","")+","+nodes[node_id].left+","+nodes[node_id].kind+","+nodes[node_id].relname+","+nodes[node_id].reltype);

            detach_source("g"+node_id.replace("n",""));

            remove_node_data(node_id);
            //Note: this node is now deleted in the serialized data model and not displayed, but these deletions are not recorded in the action protocol
        }

        nodes = parse_data();
        if (former_parent!="n0"){
            if (count_children(former_parent,nodes)==0){
                delete_node(former_parent,nodes);
            }
            else if (nodes[former_parent].kind=="span" && count_span_children(former_parent,nodes)==0){ // Span just lost its last span child, delete it
                delete_node(former_parent,nodes);
            }
            else if (nodes[former_parent].kind=="multinuc" && count_multinuc_children(former_parent,nodes)==0){ // Multinuc just lost its last multinuc child, delete it
                delete_node(former_parent,nodes);
            }
        }

    }
}

function get_depth(orig_node, probe_node, nodes){
    if (typeof nodes[probe_node] != 'undefined'){
        if (nodes[probe_node].parent != "n0"){
            parent_id = nodes[probe_node].parent;
            if (typeof nodes[parent_id] != 'undefined'){
                if (nodes[parent_id].kind != "edu" && (nodes[probe_node].relname == "span" || nodes[parent_id].kind == "multinuc" && nodes[probe_node].reltype == "multinuc")){
                    nodes[orig_node].depth += 1;
                }
            }
            get_depth(orig_node, parent_id, nodes);
        }
    }
}

function recalculate_depth(nodes){

    top_spacing = 20;
    layer_spacing = 60;
    for (node_id in nodes){
        if (nodes[node_id].kind =="edu"){
            element_id = "edu" + node_id.replace("n","");
            expected_width = 96;
        }
        else{
            element_id = "lg" + node_id.replace("n","");
            expected_width = (nodes[node_id].right - nodes[node_id].left + 1)*100 - 4;
        }
        expected_top = top_spacing + layer_spacing + nodes[node_id].depth*layer_spacing;
        expected_left = nodes[node_id].left*100 - 100;
        nid = element_id.replace(/l?g|edu/g,"n");
        relname = nodes[node_id].relname;

        if (nodes[node_id].parent=="n0"){
            detach_source(element_id.replace("l",""));
        }
        else if (typeof nodes[nodes[node_id].parent] != 'undefined'){
            parent_kind = nodes[nodes[node_id].parent].kind;
            reltype = nodes[node_id].reltype;
            if (parent_kind == "edu"){
                parent_element_id = "edu" + nodes[node_id].parent.replace("n","");
            }
            else
            {
                parent_element_id = "g" + nodes[node_id].parent.replace("n","");
            }
        }
        else{
            continue;
        }

        //Check if a new parent was initiated programmatically and needs the connection to be rendered
        if (nodes[node_id].parent!="n0" && nodes[node_id].left > 0){ //if left is 0 then this is a deleted/hidden node
            if (jsPlumb.getConnections({ source: element_id.replace("l",""), target: parent_element_id }).length <1){
                detach_source(element_id.replace("l",""));
                if (relname == "span"){
                    jsPlumb.connect({source: element_id,target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"]});
                }
                else if (parent_kind == "multinuc" && reltype=="multinuc"){
                    jsPlumb.connect({source: element_id.replace("l",""),target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(nid,"multi",relname);},location:0.2,id:"customOverlay"}]]});
                }
                else {
                    jsPlumb.connect({source: element_id.replace("l",""),target:parent_element_id, overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }], ["Custom", {create:function(component) {return make_relchooser(nid,"rst",relname);},location:0.1,id:"customOverlay"}]]});
                }
            }
        }

        sel_id = "sel" + node_id.replace("n","");
        if ($("#"+element_id.replace("l","")).length >0){ //only process elements that exist
            if (document.getElementById(element_id.replace("l","")).style.display!="none"){ //only process elements that are displayed
                if (nodes[node_id].relname != "span"){
                    if ($("#"+sel_id).length >0){
                        if (document.getElementById(sel_id).value != nodes[node_id].relname){
                            document.getElementById(sel_id).value = nodes[node_id].relname;
                        }
                    }
                    else if (nodes[node_id].relname != "none") //it's not a span, but the select is missing, reconnect
                    {
                        if (nodes[node_id].parent!="n0"){
                            detach_source(element_id);
                            if (parent_kind == "multinuc" && reltype=="multinuc"){
                                jsPlumb.connect({source: element_id.replace("l",""),target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(nid,"multi",relname);},location:0.2,id:"customOverlay"}]]});
                            }
                            else {
                                jsPlumb.connect({source: element_id.replace("l",""),target:parent_element_id, overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }], ["Custom", {create:function(component) {return make_relchooser(nid,"rst",relname);},location:0.1,id:"customOverlay"}]]});
                            }
                        }
                    }
                }
            }
        }


        if (nodes[node_id].parent == "n0"){
            detach_source(element_id.replace("l",""));
        }

        if (document.getElementById(element_id.replace("l","")).innerHTML.indexOf(nodes[node_id].left+"-"+nodes[node_id].right)<1){
            document.getElementById(element_id.replace("l","")).innerHTML = document.getElementById(element_id.replace("l","")).innerHTML.replace(/[0-9]+-[0-9]+/,nodes[node_id].left+"-"+nodes[node_id].right);
        }

        // Animate
        if (document.getElementById(element_id).style.width != expected_width + "px") { //only relevant if width has changed
            document.getElementById(element_id).style.width = expected_width + "px";
            jsPlumb.animate("wsk" + node_id.replace("n",""), {width: expected_width});
            jsPlumb.animate(element_id, {width: expected_width});
            document.getElementById(element_id.replace("l","")).style.zIndex = (200-(nodes[node_id].right - nodes[node_id].left));
        }

        if (nodes[node_id].kind !="edu" && document.getElementById(element_id.replace("l","")).style.left != nodes[node_id].anchor + "px"){
            document.getElementById(element_id.replace("l","")).style.left= nodes[node_id].anchor + "px";
            jsPlumb.animate(element_id.replace("l",""),{left: nodes[node_id].anchor});
        }

        if (nodes[node_id].kind !="edu" && document.getElementById(element_id.replace("l","")).style.top != (expected_top + 4) + "px"){
            document.getElementById(element_id.replace("l","")).style.top= (expected_top+4) + "px";
            jsPlumb.animate(element_id.replace("l",""),{top: (4+expected_top)});
            //make sure shorter units are higher in z-index
            document.getElementById(element_id.replace("l","")).style.zIndex = (200-(nodes[node_id].right-nodes[node_id].left)).toString();
        }

        if (document.getElementById(element_id).style.top != expected_top + "px" || document.getElementById(element_id).style.left != expected_left + "px"){
            jsPlumb.animate(element_id, {top: expected_top, left: expected_left});
        }
    }
}

function get_left_right(node_id, nodes, min_left, max_right){
	if (nodes[node_id].parent != "n0" && node_id != "n0" && typeof nodes[nodes[node_id].parent] != 'undefined'){
		parent = nodes[nodes[node_id].parent];
		if (min_left > nodes[node_id].left || min_left == 0){
			if (nodes[node_id].left != 0){
				min_left = nodes[node_id].left;
			}
		}
		if (max_right < nodes[node_id].right || max_right == 0){
			max_right = nodes[node_id].right;
		}
		if (nodes[node_id].relname == "span" && parent.kind =="span"){
			if (parent.left > min_left || parent.left == 0){
				parent.left = min_left;
			}
			if (parent.right < max_right){
				parent.right = max_right;
			}
		}
		else if (nodes[node_id].relname != "span"){
			if (parent.kind == "multinuc" && get_rel_type(nodes[node_id].relname) =="multinuc"){
				if (parent.left > min_left || parent.left == 0){
					parent.left = min_left;
				}
				if (parent.right < max_right){
					parent.right = max_right;
				}
			}
		}
		get_left_right(parent.id, nodes, min_left, max_right);
	}
}


function get_anchor_points(nodes){

    anchors={};
    pix_anchors={};
    sort_order = [];
    for (node_id in nodes){
        sort_order.push([node_id,nodes[node_id].depth]);
    }
    sort_order.sort(function(a, b) {return b[1] - a[1]});

    for (var i=0; i<sort_order.length; i++) {
        node_ref= sort_order[i][0];
        node = nodes[node_ref];
        if (node.kind=="edu"){
            anchors[node_ref]= "0.5";
        }
        if (node.parent!="n0" && typeof nodes[node.parent] != 'undefined'){
            parent = nodes[node.parent];
            parent_wid = (parent.right- parent.left+1) * 100 - 4;
            child_wid = (node.right- node.left+1) * 100 - 4;
            if (node.relname == "span"){
                if (node.id in anchors){
                    anchors[parent.id] = (((node.left - parent.left)*100)/parent_wid + parseFloat(anchors[node.id]) * parseFloat(child_wid / parent_wid)).toString();
                }
                else{
                    anchors[parent.id] = (((node.left - parent.left)*100)/parent_wid+(0.5*child_wid)/parent_wid).toString();
                }
            }
            else if (node.reltype=="multinuc" && parent.kind =="multinuc"){
                // For multinucs, the anchor is in the middle between leftmost and rightmost of the multinuc children
                // (not including other rst children)
                lr = get_multinuc_children_lr(node.parent, nodes);
                lr_wid = Math.floor((lr[0] + lr[1]) /2);
                lr_ids = get_multinuc_children_lr_ids(node.parent,nodes)
                left_child = lr_ids[0]
                right_child = lr_ids[1]
                if (left_child in anchors && right_child in anchors){ //both leftmost and rightmost multinuc children have been found
                    len_left = nodes[left_child].right-nodes[left_child].left+1;
                    len_right = nodes[right_child].right-nodes[right_child].left+1;
                    anchors[parent.id] = (((parseFloat(anchors[left_child]) * len_left*100 + parseFloat(anchors[right_child])* len_right * 100 + (nodes[right_child].left - parent.left) * 100)/2)/parent_wid).toString();
                }
                else{
                    anchors[parent.id] = ((lr_wid - parent.left+1) / (parent.right - parent.left+1)).toString();
                }
            }
            else{
                if (!(parent.id in anchors)){
                    anchors[parent.id] = "0.5";
                }
            }
        }
    }

    // Place anchor element to center on proportional position relative to parent, plus absolute offset to left
    for (node_id in nodes){
        node = nodes[node_id];
        pix_anchors[node_id] = (parseInt(3+node.left * 100 -100 - 39 + parseFloat(anchors[node_id])*((node.right- node.left+1) * 100 - 4))).toString();
    }


    return pix_anchors;
}

function node_exists(node_id){
nodes = parse_data();
if (node_id in nodes){
return true;
}
else
{return false;}
}

function detach_source(element_id){
    jsPlumb.select({source:element_id}).detach();
    jsPlumb.selectEndpoints({source:element_id}).each(function(ep){jsPlumb.deleteEndpoint(ep);});
    jsPlumb.repaint(element_id);
}

function detach_target(element_id){
    jsPlumb.select({target:element_id}).detach();
    jsPlumb.selectEndpoints({target:element_id}).each(function(ep){jsPlumb.deleteEndpoint(ep);});
    jsPlumb.repaint(element_id);
}

function insert_parent(node_id,new_rel,node_kind){
    nodes = parse_data();
	old_parent = nodes[node_id].parent;
	old_rel = nodes[node_id].relname;
	new_parent = "n" + (get_max_node_id(nodes) + 1);
	anchor = nodes[node_id].anchor;
    create_node_div(new_parent,nodes[node_id].depth,nodes[node_id].left,nodes[node_id].right,anchor);
	reltype = get_rel_type(old_rel);
	add_node(new_parent,old_parent,node_kind.substring(0,1),old_rel,reltype);
    if (nodes[node_id].kind=="edu"){
        child_element_id = "edu"+node_id.replace("n","");
    }
    else
    {
        child_element_id = "g"+node_id.replace("n","");
    }
    parent_element_id = "g"+new_parent.replace("n","");
    detach_source(child_element_id);

	jsPlumb.makeSource(parent_element_id, {anchor: "Top", filter: ".num_id", allowLoopback:false});
    jsPlumb.makeTarget(parent_element_id, {anchor: "Top", filter: ".num_id", allowLoopback:false});

    if (node_kind=="multinuc"){
        jsPlumb.connect({source: child_element_id,target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(node_id,"multi",new_rel);},location:0.2,id:"customOverlay"}]]});
    }
    else{ //span
        jsPlumb.connect({source: child_element_id,target:parent_element_id, connector:"Straight", anchors: ["Top","Bottom"]});
    }

    if (old_parent!="n0"){
        if (nodes[old_parent].kind=="edu"){
            old_parent_element_id = "edu"+old_parent.replace("n","");
        }
        else
        {
            old_parent_element_id = "g"+old_parent.replace("n","");
        }
        old_parent_kind = nodes[old_parent].kind;
        if (old_rel == "span"){
            jsPlumb.connect({source: parent_element_id,target:old_parent_element_id, connector:"Straight", anchors: ["Top","Bottom"]});
        }
        else if (old_parent_kind == "multinuc" && reltype=="multinuc"){
            jsPlumb.connect({source: parent_element_id,target:old_parent_element_id, connector:"Straight", anchors: ["Top","Bottom"], overlays: [ ["Custom", {create:function(component) {return make_relchooser(new_parent,"multi",old_rel);},location:0.2,id:"customOverlay"}]]});
        }
        else {
            jsPlumb.connect({source: parent_element_id,target:old_parent_element_id, overlays: [ ["Arrow" , { width:12, length:12, location:0.95 }], ["Custom", {create:function(component) {return make_relchooser(new_parent,"rst",old_rel);},location:0.2,id:"customOverlay"}]]});
        }
    }

    //Update rel before updating parent, since reltype is important for left-right calculation
    //Do this last because the select overlay to update will not necessarily exist before connections are made
	update_rel(node_id,new_rel,nodes);
	update_parent(node_id,new_parent);
	nodes = parse_data();
	modify_undo("+qrl:"+node_id.replace("n","")+","+nodes[node_id].relname+";",";"); //qrl is spurious for insert_parent, remove it

}

function get_max_node_id(nodes){
    max = 0;
    for (node_id in nodes){
        node_num = parseInt(node_id.replace("n",""));
        if (node_num > max){
            max = node_num;
        }
    }
    return max;
}

function add_node(node_id,parent,node_kind_abbr,relname,reltype){
    document.getElementById("data").value = document.getElementById("data").value + ";" + node_id + "," + parent +","+node_kind_abbr+",0,"+relname+","+reltype;
}

function create_node_div(id,depth,left,right,anchor){

id = id.replace("n","");

if ($('#lg' + id).length > 0){ // A node with this id already exists, just display it
    document.getElementById("lg"+id).style.display = "block";
    document.getElementById("g"+id).style.display = "block";
    document.getElementById("g"+id).style.zIndex = (200-(right-left));
    expected_left = left*100 - 100+"px";
    document.getElementById("g"+id).style.left = expected_left;
    jsPlumb.recalculateOffsets("g"+id);
    jsPlumb.repaint("g"+id);
    document.getElementById("lg"+id).style.left = expected_left;
    jsPlumb.recalculateOffsets("lg"+id);
    jsPlumb.repaint("lg"+id);

    return;
}

width = right - left + 1;
top_spacing = 20;
layer_spacing = 60;
var lg = document.createElement("div");
lg.setAttribute("id", "lg" + id);
lg.className = "group";
lg.style.left = (left*100 - 100) + "px";
lg.style.top = (top_spacing + layer_spacing+depth*layer_spacing) + "px";
lg.style.width =(width *100 -4) + "px";
lg.innerHTML = '<div id="wsk'+id+'" class="whisker" style="width:'+(width *100 -4) + 'px;"></div></div>';

if (document.getElementById("use_span_buttons").value=="True"){
    use_span_buttons = true;
}
else{
    use_span_buttons = false;
}
if (document.getElementById("use_multinuc_buttons").value=="True"){
    use_multinuc_buttons = true;
}
else{
    use_multinuc_buttons = false;
}

var g = document.createElement("div");
g.setAttribute("id", "g" + id);
g.className = "num_cont";
g.style.left = (anchor)+"px";
g.style.top = (4+top_spacing + layer_spacing+depth*layer_spacing)+'px';
g.style.position = 'absolute';
g.style.zIndex = (200-(right-left)).toString();
innerHTML_string = '<table class="btn_tb"><tr><td rowspan="2"><button id="unlink_'+ id+'" class="minibtn" onclick="act('+"'up:"+id+",0'"+');">X</button></td><td rowspan="2"><span class="num_id">'+left+"-"+right+'</span></td>';
if (use_span_buttons){
    innerHTML_string += '<td><button id="aspan_'+ id+'" class="minibtn" onclick="act('+"'sp:"+id+"'"+');">T</button></td>';
}
innerHTML_string += '</tr>';
if (use_multinuc_buttons){
    innerHTML_string += '<tr><td><button id="amulti_'+ id+'" class="minibtn" onclick="act('+"'mn:"+id+"'"+');">Λ</button></td></tr>';
}
innerHTML_string += '</table>';
g.innerHTML = innerHTML_string;

document.getElementById("inner_canvas").appendChild(lg);
document.getElementById("inner_canvas").appendChild(g);

jsPlumb.recalculateOffsets("g"+id);
jsPlumb.repaint("g"+id);

}

function get_multinuc_children_lr(multinuc_id,nodes){
    right = 0;
    left = 10000;
    for (node_id in nodes){
        if (nodes[node_id].parent == multinuc_id && nodes[node_id].reltype =="multinuc"){
            if (nodes[node_id].left < left){left = nodes[node_id].left;}
            if (nodes[node_id].right > right){right = nodes[node_id].right;}
        }
    }
	return [left,right];
}

function get_multinuc_children_lr_ids(multinuc_id,nodes){
    right = 0;
    left = 10000;
    node_right = "";
    node_left = "";
    for (node_id in nodes){
        if (nodes[node_id].parent == multinuc_id && nodes[node_id].reltype =="multinuc"){
            if (nodes[node_id].left < left){left = nodes[node_id].left; node_left = node_id;}
            if (nodes[node_id].right > right){right = nodes[node_id].right; node_right = node_id;}
        }
    }
	return [node_left,node_right];
}

function is_ancestor(new_parent_id,node_id){

    // check if node_id is an ancestor of new_parent
    nodes=parse_data();
    parent = nodes[new_parent_id].parent;
    while (parent != "n0"){
        if (parent == node_id){
            return true;
        }
        parent = nodes[parent].parent;
    }
    return false;
}