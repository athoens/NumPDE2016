function [e,eIndex,boundaryNodes,boundaryEdges]=edgeMatrix(p,t)
%
% collects indices of edges, builds a mapping from node indices to edge
% indices and identifies the boundary nodes and edges within a mesh
%
% input:
% p - Nx2-array of nodal coordinates
% t - Mx3-array of triangles as indices into p, defined with a 
%     counter-clockwise node ordering
%
% output:
% e             - Ex2-array of edge node correspondence
% eIndex        - NxN-sparse matrix of all node combinations as indices
%                 into e
% boundaryNodes - vector of all node indices that lie on the boundary
% boundaryEdges - vector of all edge indices that lie on the boundary


% number of vertices
N = size(p,1);

% initialize sparse matrices for edge indices and edge appearance
eIndex  = sparse(N,N);
eAppear = sparse(N,N);

% indices of the nodes of all edges of a triangle
elementEdge = [[1,2];[2,3];[3,1]];

% edge index
edgeIndex = 1;

% initialize edge2node matrix e with maximum length
e = zeros(3*size(t,1),2);

% loop through all triangles
for i=1:size(t,1)
    % loop trough all edges of the triangle
    for j=1:3
        % read nodes of edge in ascending order
        node1 = min(t(i,elementEdge(j,:)));
        node2 = max(t(i,elementEdge(j,:)));
        
        % write appearance matrix (where only upper triangular matrix has
        % entries)
        eAppear(node1,node2) = eAppear(node1,node2) + 1;
        
        % write edge index matrix and edge2node matrix
        if eAppear(node1,node2) == 1
            eIndex(node1,node2) = edgeIndex;
            eIndex(node2,node1) = edgeIndex;
            e(edgeIndex,1) = node1;
            e(edgeIndex,2) = node2;
            edgeIndex = edgeIndex + 1;
        end
    end
end

% delete empty rows of e
e = e(1:edgeIndex-1,:);

% find boundary nodes and edges
[beRow,beCol] = find(eAppear==1);
boundaryNodes = unique([beRow;beCol]);
boundaryEdges = unique(full(diag(eIndex(beRow,beCol))));