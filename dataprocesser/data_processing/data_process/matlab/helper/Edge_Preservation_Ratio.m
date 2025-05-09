function [EPR,EdgeGenRatio] = Edge_Preservation_Ratio(ref,img)
% calculates the edge preservation ratio (EPR) of a reference and an image
% and returns a true bool variable, if additional edges are created
BW1 = edge(ref,'Canny'); % computational expensive
BW2 = edge(img,'Canny');
overlap = BW1.*BW2; % calculate the overlap of both edge images
nonZeroRef = nnz(BW1); % calculates number of nonzero elements
nonZeroImg = nnz(BW2);
nonZeroOverlap = nnz(overlap);
EPR = nonZeroOverlap/nonZeroRef;
EdgeGenRatio = nonZeroImg/nonZeroRef;
end

