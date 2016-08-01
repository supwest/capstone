
$(function() {
	console.log('jq is working');
	createVis();
    //createDend();
    //createVis2();
});

function createVis() {

//var width=960;
//var height = 700;


//var svg = d3.select("#chart").append("svg").attr("width", width).attr("height", height);
	
	var svg = d3.select("svg"),
		width = +svg.attr("width"),
		height = +svg.attr("height");
	
var color=d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()
	.force("link", d3.forceLink().id(function(d) { return d.id; }).strength(function(d) { return Math.sqrt(d.value)/150; }))
	.force("charge", d3.forceManyBody().strength(function() { return -70; }))
	.force("center", d3.forceCenter(width / 2, height / 2));
	
d3.json("/data", function(error, graph) {
	if (error) throw error;
	
	var link = svg.append("g")
		.attr("class", "links")
		.selectAll("line")
		.data(graph.links)
		.enter().append("line")
			.attr("stroke-width", function(d) { return d.value; });
			
	var node = svg.append("g")
		.attr("class", "links")
		.selectAll("circle")
		.data(graph.nodes)
		.enter().append("circle")
		.attr("r", 5)
		.attr("fill", function(d) { return color(d.group); })
		.call(d3.drag()
			.on("start", dragstarted)
			.on("drag", dragged)
			.on("end", dragended));
			
	node.append("title")
		.text(function(d) { return d.id; });
		
	simulation
		.nodes(graph.nodes)
		.on("tick", ticked);
		
	simulation.force("link")
		.links(graph.links);
		
	function ticked() {
	link
		.attr("x1", function(d) { return d.source.x; })
		.attr("y1", function(d) { return d.source.y; })
		.attr("x2", function(d) { return d.target.x; })
		.attr("y2", function(d) { return d.target.y; });
		
	node
		.attr("cx", function(d) { return d.x; })
		.attr("cy", function(d) { return d.y; })
	}
});

function dragstarted(d) {
	if (!d3.event.active) simulation.alphaTarget(0.3).restart();
	d.fx = d.x;
	d.fy = d.y;
}

function dragged(d) {
	d.fx = d3.event.x;
	d.fy = d3.event.y;
}

function dragended(d) {
	if (!d3.event.active) simulation.alphaTarget(0);
	d.fx = null;
	d.fy = null;
}
}

//function createDend() {
//    var svg = d3.select("svg"),
//    width = +svg.attr("width"),
//    height = +svg.attr("height"),
//    g = svg.append(g).attr("transform", "translate(40,0)");
//
//    var tree = d3.cluster().size([height, width = 160]);
//
//    var stratify = d3.stratify().parentId(function(d) { return d.id.substring(0m 

//}
//

function createVis2() {
    var width = 960,
    height = 500,
    padding = 1.5,
    clusterPadding = 6,
    maxRadius = 12;

    var num_clusters = 4;

    var color = d3.scale.category10()
        .domain(d3.range(num_clusters));

    var clusters = new Array(num_clusters);

    var svg = d3.select("body").append("svg").attr("width", width)
        .attr("height", height);

    d3.json('/data2', function (error, graph) {

        d3.layout.pack().sort(null).size([width, height])
            .children(function(d) { return d.values; })
            .value(function(d) { return d.rating; })
            .nodes({values: d3.nest()
                .key(function(d) {return d.cluster; })
                .entries(graph.nodes)});

        var force = d3.layout.force()
            .nodes(graph.nodes)
            .size([width, height])
            .gravity(.02)
            .charge(0)
            .on("tick", tick)
            .start();

        var node = svg.selectAll("circle")
            .data(graph.nodes)
            .enter().append("circle")
            .style("fill", function(d) { return color(d.cluster); })
            .call(force.drag);

        node.transition()
            .duration(750)
            .delay(function(d, i) {return i*5})
            .attrTween("r", function(d) {
                var i =d3.interpolate(0, d.radius);
                return function(t) {return d.radius = i(t); };
            })'=;

    }

    function tick(e) {
        node.each(cluster(10 * e.alpha * e.alpha))
            .each(collide(.5))
            .attr("cx", function(d) { return d.x })
            .attr("cy", function(d) { return d.y });
    }

    function cluster(alpha) {
       return function(d) {
          var cluster = clusters[d.cluster]
          if (cluster === d) return;
          var x = d.x - cluster.x,
          y = d.y - cluster.y,
          l = Math.sqrt(x * x + y * y),
          r = d.radius + cluster.radius
          if (l != r) {
            l = (l - r) / l * alpha;
            d.x -= x *= l;
            x.y -= y *= l;
            cluster.x += x;
            cluster.y += y;
          }
    }; 
}

    function collide(alpha) {
        var quadtree = d3.geom.quadtree(nodes);
        return function (d) {
            var r = d.radius + maxRadius + Math.max(padding, clusterPadding),
                nx1 = d.x - r,
                nx2 = d.x + r,
                ny1 = d.y - r,
                ny2 = d.y + r;
            quadtree.visit(function(quad, x1, y1, x2, y2) {
                if (quad.point && (quad.point !== d)) {
                    var x = d.x - quad.point.x,
                        y = d.y - quad.point.y,
                        l = Math.sqrt(x*x + y*y),
                        r = d.radius + quad.point.radius + (d.cluster === quad.point.cluster ? padding : clusterPadding);
                    if (l < r) {
                        l = (l - r) / l * alpha;
                        d.x -= x *= l;
                        d.y -= y *= l;
                        quad.point.x += x;
                        quad.point.y += y;
                    }
                }
                return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
            });
    };
    }

}
