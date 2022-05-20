"""
Diese Methode wurde nach 
[Mechrez, R., Shechtman, E. & Zelnik-Manor, L. Saliency driven image manipulation. Machine Vision and Applications 30, 189–202 (2019).]
komplett selbst Programmiert.
Das Vorgehen kann im Paper nachgelesen werden. Es wurden ein paar Anpassungen gemacht um die Laufzeit erträglich zu machen ohne das an Qualität maßgeblich gespart wurde. 
"""
import numpy as np

try:
    import pycuda
    pycuda_installed = True
except ModuleNotFoundError:
    print('pycuda not installed, trying CPU fallback')
    pycuda_installed = False
# Distinguish between CPU and GPU computation
if pycuda_installed:
        from PatchMatchCuda import PatchMatch
        tau_plus = 0.4
        tau_minus = 0.6
else:
        from PatchMatchOrig import PatchMatch
        tau_minus = 0.7
        tau_plus = 0.48

import cv2
import matplotlib.pyplot as plt
from NN.inference import get_predictions
from project.saliency import static_fine_saliency_core
import heapq
from util import VideoReader


class Recoloring:
    # values needed for calculating optimal thresholds in Database_Update
    nu = 0.1
    delta_S = 0.6
    epsilon = 0.05
    stop_iterating = False

    tau_minus = 0.7
    tau_plus = 0.48

    def Image_Update(self, tau_plus_Database, tau_minus_Database, img_field, trg_plus, trg_minus, bw_msk_in_R_field, bw_msk_out_R_field):
        # The PatchMatch Algorithm needs to be executed twice for each iteration (for each Database once to enhance the area in trg_plus and diminish the area in trg_minus)
        pm = PatchMatch(trg_plus, trg_plus, tau_plus_Database, tau_plus_Database) # Execute Patchmatch Algorithm
        pm.propagate(iters=7,rand_search_radius=224)
        trg_plus = pm.reconstruct_avg(tau_plus_Database, 1) # Replace patches from img with patches from first entry PatchMatch

        pm = PatchMatch(trg_minus, trg_minus, tau_minus_Database, tau_minus_Database) # Execute Patchmatch Algorithm
        pm.propagate(iters=7,rand_search_radius=224)
        trg_minus = pm.reconstruct_avg(tau_minus_Database, 1) # Replace patches from img with patches from first entry PatchMatch

        # get the images to normal RGB format otherwise cv.seemlessClone would not work
        img_field = (img_field*255).astype(np.uint8)
        trg_plus = (trg_plus*255).astype(np.uint8)
        trg_minus = (trg_minus*255).astype(np.uint8)

        img_field = cv2.cvtColor(img_field, cv2.COLOR_LAB2RGB)
        trg_plus = cv2.cvtColor(trg_plus, cv2.COLOR_LAB2RGB)
        trg_minus = cv2.cvtColor(trg_minus, cv2.COLOR_LAB2RGB)

        trg_gray = trg_plus.copy()
        trg_gray = cv2.cvtColor(trg_gray, cv2.COLOR_RGB2GRAY)
        (thresh, trg_gray) = cv2.threshold(trg_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # apply errosion to get rid of edge pixel which will cause weird looking fragments if not eroded
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
        eros = cv2.erode(bw_msk_in_R_field, element)

        cut_out = cv2.bitwise_and(img_field, img_field, mask = eros)
        cut_out = cv2.bitwise_and(cut_out, cut_out, mask = ~trg_gray)
        trg_minus = cv2.bitwise_and(trg_minus, trg_minus, mask = ~bw_msk_in_R_field)
        # combine everything to one picture
        trg_plus = cut_out + trg_plus + trg_minus
        # apply seemless clone which will apply the above created picture to the original 
        trg_plus = cv2.seamlessClone(trg_plus, img_field, bw_msk_in_R_field + bw_msk_out_R_field, (int(trg_plus.shape[0]/2), int(trg_plus.shape[1]/2)), flags=cv2.MIXED_CLONE)
        trg_plus = cv2.seamlessClone(trg_minus, trg_plus, bw_msk_out_R_field, (int(trg_plus.shape[0]/2), int(trg_plus.shape[1]/2)), flags=cv2.NORMAL_CLONE)

        trg_plus = cv2.cvtColor(trg_plus, cv2.COLOR_RGB2LAB)
        trg_plus = (trg_plus/255).astype(np.float32) # convert back to operating format for the next iterations
    

        return trg_plus



    def Database_Update(self, img_field ,img_saliency_field, trg_field, bw_msk_in_R_field, bw_msk_out_R_field, tau_plus_i, tau_minus_i, sol_tau_plus, sol_tau_minus, smallest_lemma):

        # convert so it can be used for saliency computation
        trg_field = (trg_field*255).astype(np.uint8)
        trg_field = cv2.cvtColor(trg_field, cv2.COLOR_LAB2RGB)
        # get salient image of color corrected image (the first iteration of each scale will be the original image)
        trg_saliency, threshmap = static_fine_saliency_core(trg_field) # generates salient picture in BGR
        trg_saliency = cv2.cvtColor(trg_saliency, cv2.COLOR_BGR2RGB)
        # convert back 
        trg_field = cv2.cvtColor(trg_field, cv2.COLOR_RGB2LAB)
        trg_field = (trg_field/255).astype(np.float32)



        trg_saliency_in_R = cv2.bitwise_and(trg_saliency, trg_saliency, mask=bw_msk_in_R_field) # generate image with salient pixel only in mask (inside RoI)
        trg_saliency_out_R = cv2.bitwise_and(trg_saliency, trg_saliency, mask=bw_msk_out_R_field) # generate image with salient pixel only in mask (outside RoI)

        tau_plus_Database = np.uint8(np.zeros(img_field.shape)) # Generate Zero Matrices in shape of img_field which will hold the areas of salient pixel >= tau_plus
        tau_minus_Database = np.uint8(np.zeros(img_field.shape)) # Generate Zero Matrices in shape of img_field which will hold the areas of salient pixel <= tau_minus

        tau_plus_Database = cv2.cvtColor(tau_plus_Database, cv2.COLOR_RGB2LAB)
        tau_minus_Database = cv2.cvtColor(tau_minus_Database, cv2.COLOR_RGB2LAB)

        tau_plus_Database = (tau_plus_Database/255).astype(np.float32)
        tau_minus_Database = (tau_minus_Database/255).astype(np.float32)

        trg_plus = tau_plus_Database.copy()
        trg_minus = tau_minus_Database.copy()

        # iterate over each pixel in the Colorblind image (each image has the same size anyway)
        for x in range(img_field.shape[0]):
            for y in range(img_field.shape[1]):

                if bw_msk_in_R_field[x, y] == 255: # get the cut out version of the Colorblind image (wil contain the pixel inside the original color image Saliency Map)
                    trg_plus[x, y] = img_field[x, y]
                elif bw_msk_out_R_field[x, y] == 255: # inverse of the step above
                    trg_minus[x, y] = img_field[x, y]

                if img_saliency_field[x, y] >= tau_plus_i:
                    tau_plus_Database[x, y] = img_field[x, y] # pixel assignment if >= tau_plus (Generates image with only most salient area)
                elif img_saliency_field[x, y] <= tau_minus_i:
                    tau_minus_Database[x, y] = img_field[x, y] # pixel assignment if <= tau_plus (Generates image with only least salient area)

        # add 20% of the most Salient pixel 
        sum_in_R = heapq.nlargest(int((trg_saliency_in_R.shape[0] * trg_saliency_in_R.shape[0] * 20) / 100), trg_saliency_in_R.flatten())
        mean_in_R = np.mean(sum_in_R)
        # add 20% of the least Salient pixel 
        sum_out_R = heapq.nlargest(int((trg_saliency_out_R.shape[0] * trg_saliency_out_R.shape[0] * 20) / 100), trg_saliency_out_R.flatten())
        mean_out_R = np.mean(sum_out_R)

        # the above mentioned Paper sugessts an Equation to update the thresholds tau_plus and tau_minus
        # tau_plus_i+1 = tau_plus_i + nu * ||lemma_in_R - delta_S||
        # tau_minus_i+1 = tau_minus_i - nu * ||lemma_out_R - delta_S||

        lemma_in_R = mean_in_R - mean_out_R
        lemma_out_R = mean_out_R - mean_in_R

        tau_plus = tau_plus_i + (self.nu * abs(lemma_in_R - self.delta_S))
        tau_minus = tau_minus_i - (self.nu * abs(lemma_out_R - self.delta_S))

        if abs(lemma_in_R - self.delta_S) < smallest_lemma: # threshold to determine optimal tau value
            sol_tau_plus = tau_plus_i
            sol_tau_minus = tau_minus_i
            smallest_lemma = abs(lemma_in_R - self.delta_S)
        # Condition to stop iterating if tau values dont change anymore (will probably never happen)
        if np.round(tau_plus_i,2) == np.round(tau_plus,2) or np.round(tau_minus_i,2) == np.round(tau_minus,2):
            print("HALT")
            stop_iterating = True
        # Condition to stop iterating if threshold is met (will probably never happen)
        elif abs(lemma_in_R - self.delta_S) < self.epsilon:
            print("HALT")
            stop_iterating = True

        return trg_plus, trg_minus, tau_plus_Database, tau_minus_Database, tau_plus, tau_minus, sol_tau_plus, sol_tau_minus, smallest_lemma

    def recolor_core(self, img_normal, img_cvd):
        img_cvd = cv2.cvtColor(img_cvd, cv2.COLOR_BGR2RGB)  # For convenience convert to RGB image
        img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)  # For convenience convert to RGB image

        bw_mask_in_R = get_predictions(
            img_normal)  # get saliency mask from normal color image used to cut out salient area

        bw_mask_in_R = (bw_mask_in_R * 255).astype(np.uint8)  # preparation for non salient area masking

        bw_mask_out_R = ~bw_mask_in_R  # swap black and white area (non salient area masking)

        # different versions for GPU and CPU implementation
        if pycuda_installed:  # normal implementation with higher resolution (higher resolution = higher computation time)
            itter = 10
            img = cv2.resize(img_cvd, (512, 512))
            trg = cv2.resize(img, (512, 512))
            bw_mask_in_R = cv2.resize(bw_mask_in_R, (512, 512))
            bw_mask_out_R = cv2.resize(bw_mask_out_R, (512, 512))

        else:  # light weight version with less resolution
            img = cv2.resize(img_cvd, (256, 256))
            trg = cv2.resize(img, (256, 256))
            bw_mask_in_R = cv2.resize(bw_mask_in_R, (256, 256))
            bw_mask_out_R = cv2.resize(bw_mask_out_R, (256, 256))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        trg = cv2.cvtColor(trg, cv2.COLOR_RGB2LAB)

        img_saliency, thresh_map = static_fine_saliency_core(
            img)  # compute saliency map of Colorblind Image (The algorithm expects what the Colorblind Salient map is different then the normal image Salient Map)

        img = (img / 255).astype(np.float32)  # Algorithm works with Matrices with values between 0 and 1
        trg = (trg / 255).astype(np.float32)

        # create image fields because we want to keep the original matrices and we wanna avoid using cv.pyrUp after cv.pyrDown (causes Blurring)
        img_field = [img]
        trg_field = [trg]
        img_saliency_field = [img_saliency]
        bw_msk_in_R_field = [bw_mask_in_R]
        bw_msk_out_R_field = [bw_mask_out_R]

        for i in range(10):  # downsample every image
            n = i

            img_field.append(cv2.pyrDown(img_field[i]))
            trg_field.append(cv2.pyrDown(trg_field[i]))
            img_saliency_field.append(cv2.pyrDown(img_saliency_field[i]))
            bw_msk_in_R_field.append(cv2.pyrDown(bw_msk_in_R_field[i]))
            bw_msk_out_R_field.append(cv2.pyrDown(bw_msk_out_R_field[i]))

            if img_field[i + 1].shape[0] < 100 or img_field[i + 1].shape[
                1] < 100:  # break if a certain size is reached
                break

        sol_tau_plus = 0.2  # start optimal value will be adjusted for every iteration
        sol_tau_minus = 0.8  # start optimal value will be adjusted for every iteration
        smallest_lemma = 2  # start optimal value will be adjusted for every iteration
        itter = None
        if pycuda_installed == False:  # change to smaller iteration count higher resolution for lesser computaion time
            n = 0
            itter = 1

        for i in range(n, -1,
                       -1):  # go the image fields down (for every iteration a image with higher resolution till original resolution)
            for j in range(itter):  # numbers of iterations per resolution

                if self.tau_plus >= 0.85:  # dont cross this threshold closer to 1 will cause recreating the original image without color correction
                    tau_plus = sol_tau_plus  # if threshold is reached change tau to optimal value
                if self.tau_minus <= 0.2:
                    tau_minus = sol_tau_minus
                if self.stop_iterating == True or (self.tau_plus >= 0.85 and self.tau_minus <= 0.2):  # stop iterating is true if
                    break
                # Categorize the salient image of the Colorblind Image by applying the thresholds tau_plus and tau_minus creating two images (tau_Database) containing
                # only the salient area or the non salient area (save the optimal tau values for the steps above)
                trg_plus, trg_minus, tau_plus_Database, tau_minus_Database, tau_plus, tau_minus, sol_tau_plus, sol_tau_minus, smallest_lemma = self.Database_Update(
                    img_field[i], img_saliency_field[i], trg_field[i], bw_msk_in_R_field[i], bw_msk_out_R_field[i],
                    self.tau_plus, self.tau_minus, sol_tau_plus, sol_tau_minus, smallest_lemma)

                # Get a recolor Colorblind image
                trg_field[i] = self.Image_Update(tau_plus_Database, tau_minus_Database, img_field[i], trg_plus,
                                                 trg_minus, bw_msk_in_R_field[i], bw_msk_out_R_field[i])

            # after each scale start at the otimal tau values to search in the next iterationts near them to find even better values
            self.tau_plus = sol_tau_plus
            self.tau_minus = sol_tau_minus

            if trg_field[i - 1].shape[1] > 200 and trg_field[i - 1].shape[
                1] < 500 and pycuda_installed == True:  # change number of iterations for spcified scale to reduce computation time (optimal values can be found at lower resolutionts as well)
                itter = int(itter / 3)
            elif trg_field[i - 1].shape[1] > 500 and pycuda_installed == True:
                itter = 1
            elif pycuda_installed == True:
                itter = 5
            elif pycuda_installed == False:  # upsample only for CPU calculation to get OK Solutions
                trg_field[i] = cv2.pyrUp(trg_field[i], dstsize=(512, 512))

        trg_field[0] = (trg_field[0] * 255).astype(np.uint8)
        trg_field[0] = cv2.cvtColor(trg_field[0], cv2.COLOR_LAB2RGB)  # solution
        return trg_field

    def recolor(self, img_normal, img_cvd):
        if __name__ == "__main__":
            recolor = Recoloring(None)

            trg_field = recolor.recolor_core(img_normal, img_cvd)

            plt.imshow(trg_field[0])
            plt.show()

    def __init__(self, transformer):
        self.transformer = transformer
        self.reader = transformer.reader

    def transform(self):
        for original, transformed in self.transformer.transform():
            #if np.max(original) == 0 or np.max(transformed) == 0:
            #    # Algorithm doesn't support 0-arrays/black images, return accordingly shaped 0-array
            #    yield original, np.repeat(np.uint64(0), original.shape[0] * original.shape[1]).reshape(original.shape[0], -1)
            yield original, self.recolor_core(original, transformed)[0]  # needs normal color image and colorblind image
        
    @property
    def provides_type(self):
        return 'rgb888'

    def provides_shape(self):
        """Tuple of (width, height) returned by Recolor"""
        return (512, 512)

