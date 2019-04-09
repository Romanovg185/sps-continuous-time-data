library(dplyr)
library(igraph)
library(arcdiagram)

# EDGE DOES NOT REFER TO ID, BUT TO NODE POSITION!!!

#mis_graph = read.graph("/home/romano/mep/ContinuousGlobalSynchrony/Graphs/1_4000_fr_16082018_161839.gml", format="gml")
mis_graph = read.graph("/home/romano/Desktop/arcdiagrammaster/lesmiserables.gml")

edgelist = get.edgelist(mis_graph)

#arcplot(edgelist)

#arcplot(edgelist, cex.labels=0.8,
#        show.nodes=TRUE, lwd.nodes = 2, line=-0.5, 
#        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 1.5)

# what's in mis_graph
#mis_graph

vlabels = get.vertex.attribute(mis_graph, "label")
# get vertex groups
vgroups = get.vertex.attribute(mis_graph, "group")
# get vertex fill color
vfill = get.vertex.attribute(mis_graph, "fill")
# get vertex border color
vborders = get.vertex.attribute(mis_graph, "border")
# get edges value
values = get.edge.attribute(mis_graph, "value")
# get vertex degree
degrees = degree(mis_graph)



q <- arcplot(edgelist, labels=vlabels, cex.labels=0.8,
        show.nodes=TRUE, col.nodes=vborders, bg.nodes=vfill, 
        cex.nodes = log(degrees)+0.5, pch.nodes=21,
        lwd.nodes = 2, line=-0.5, 
        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 1.5 * values)

#arcplot(edgelist, labels=vlabels, cex.labels=0.8,
#        show.nodes=TRUE, col.nodes=vborders, bg.nodes=vfill, 
#        cex.nodes = log(degrees)+0.5, pch.nodes=21,
#        lwd.nodes = 2, line=-0.5, 
#        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 1.5 * values)

# data frame with node attributes
#x = data.frame(vgroups, degrees, vlabels, ind=1:vcount(mis_graph))
#y = arrange(x, desc(vgroups), desc(degrees))
#new_ord = y$ind
#arcplot(edgelist, ordering=new_ord, labels=vlabels, cex.labels=0.8,
#        show.nodes=TRUE, col.nodes=vborders, bg.nodes=vfill, 
#        cex.nodes = log(degrees)+0.5, pch.nodes=21,
#        lwd.nodes = 2, line=0, 
#        col.arcs = hsv(0, 0, 0.2, 0.25), lwd.arcs = 1.5 * values)
