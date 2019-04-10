library(dplyr)
library(igraph)
library(arcdiagram)
library(hashmap)
# EDGE DOES NOT REFER TO ID, BUT TO NODE POSITION!!!

mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/scale_degree.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/1_4000_fr_16082018_161839_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_120304_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_121719_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_123148_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_131530_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_132308_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_132838_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_152235_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_152829_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_153856_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_161337_cross.gml", format="gml")
#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/16082018_162452_cross.gml", format="gml")

edgelist = get.edgelist(mis_graph)
node_ordering <- graph_info(edgelist)$nodes
real_ordering <- graph_info(edgelist)$aux_ord
mappy <- hashmap(node_ordering, real_ordering)
#arcplot(edgelist, cex.labels=0.8, sorted=TRUE,
#        show.nodes=TRUE, lwd.nodes = 2, line=-0.5,
#        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 1.5)
# what's in mis_graph
#mis_graph
vlabels <- get.vertex.attribute(mis_graph, "label")
vrlabels <- vlabels
for(i in 1:length(vlabels)){
vrlabels[mappy[[i]]] = vlabels[i]
}
# get vertex groups
vgroups = get.vertex.attribute(mis_graph, "group")
vrgroups <- vgroups
for(i in 1:length(vgroups)){
vrgroups[mappy[[i]]] = vgroups[i]
}
# get vertex fill color
vfill = get.vertex.attribute(mis_graph, "fill")
vrfill <- vfill
for(i in 1:length(vfill)){
vrfill[mappy[[i]]] = vfill[i]
}
# get vertex border color
vborders = get.vertex.attribute(mis_graph, "border")
vrborders <- vborders
for(i in 1:length(vborders)){
vrborders[mappy[[i]]] = vborders[i]
}
# get edges value
values = get.edge.attribute(mis_graph, "value")
vralues <- values
for(i in 1:length(values)){
vralues[mappy[[i]]] = values[i]
}
# get vertex degree
degrees = degree(mis_graph)
dregrees <- degrees
for(i in 1:length(degrees)){
dregrees[mappy[[i]]] = degrees[i]
}

# data frame with node attributes
x = data.frame(vrgroups, dregrees, vrlabels, ind=1:vcount(mis_graph))
y = arrange(x, desc(vrgroups), desc(dregrees))
new_ord = y$ind
arcplot(edgelist, ordering=new_ord, labels=vrlabels, cex.labels=0.8,
        show.nodes=TRUE, col.nodes=vrborders, bg.nodes=vrfill,
        cex.nodes = log(dregrees)+0.5, pch.nodes=21,
        lwd.nodes = 2, line=0,
        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 0.2 * values^2)
