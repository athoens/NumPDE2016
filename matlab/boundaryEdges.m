function be=boundaryEdges(t)
%
% identifies the boundary edges within a mesh
%
% input:
% p - array of nodal coordinates
% t - array of triangles as indicies into p, defined with a 
%     counter-clockwise node ordering
%
% output:
% be - Bx2-array of edges as indicies into p where B is the number of
%      boundary edges

% initialize b as vector with maximum length (i.e. number of all traingles 
% times three)
b = zeros(3*size(t,1),3);

% variable to count the identified boundary edges
b_index = 0;

% indices of the nodes of all edges of a triangle
elementEdge = [[1,2];[2,3];[3,1]];

% loop through all triangles
for i=1:size(t,1)
    % loop trough all edges of the triangle
    for j=1:3
        % read nodes in ascending order
        node1 = min(t(i,elementEdge(j,:)));
        node2 = max(t(i,elementEdge(j,:)));
        % set boolean that indicates whether edge is new to true
        newEdge = true;
        % initialize counter
        k = 1;
        % loop through all edges that have been found
        while ((k <= b_index) && newEdge)
            if (b(k,1) == node1 && b(k,2) == node2)
                % edge exists already, increment appearance counter
                newEdge = false;
                b(k,3)  = b(k,3) + 1;
            end
            k = k+1;
        end
        if newEdge
            % edge does not exist yet, create new edge
            b_index = b_index + 1;
            b(b_index,1) = node1;
            b(b_index,2) = node2;
            b(b_index,3) = 1;
        end
    end
end

% delete all empty rows of b
b = b(1:b_index,:);

% create new array only containing edges with single appearance
be = b(b(:,3)==1,1:2);
% zeros(b_index,2);
% be_index = 0;
% % loop through all rows of edge-array
% for i=1:b_index
%     if b(i,3) == 1
%         % edge appeared only once, write edge to new array
%         be_index = be_index + 1;
%         be(be_index,:) = b(i,1:2);
%     end
% end
% % delete all empty rows of b
% be = be(1:be_index,:);
% end
