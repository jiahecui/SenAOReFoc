            # Divide each search block into 16 elements (4 in each dimension)
            for i in range(len(self.act_ref_cent_coord)):
    
                # Get reference centroid coords of each element
                elem_ref_cent_coord_x = np.arange(self.act_ref_cent_coord_x[i] - self.SB_rad + self.elem_size // 2 , \
                    self.act_ref_cent_coord_x[i] + self.SB_rad - self.elem_size // 2, self.elem_size)
                elem_ref_cent_coord_y = np.arange(self.act_ref_cent_coord_y[i] - self.SB_rad + self.elem_size // 2 , \
                    self.act_ref_cent_coord_y[i] + self.SB_rad - self.elem_size // 2, self.elem_size)
                
                print('X coords are: {}\nY coords are: {}'.format(elem_ref_cent_coord_x, elem_ref_cent_coord_y))

                # Get 1D coords of each element reference centroid
                elem_ref_cent_coord = np.zeros(self.div_elem * self.div_elem, dtype = int)

                for j in range(self.div_elem):
                    for k in range(self.div_elem):
                        elem_ref_cent_coord[j * self.div_elem + k] = elem_ref_cent_coord_y[j] * \
                            self.sensor_width + elem_ref_cent_coord_x[k]

                # Calculate centroids for each element in one search block
                for j in range(len(elem_ref_cent_coord)):

                    # Initialise temporary summing parameters
                    sum_x = 0
                    sum_y = 0
                    sum_pix = 0

                    # Get 2D coords of pixels in each element that need to be summed
                    elem_pix_coord_x = np.arange(elem_ref_cent_coord_x[j % 4] - self.elem_rad, \
                        elem_ref_cent_coord_x[j % 4] + self.elem_rad)
                    elem_pix_coord_y = np.arange(elem_ref_cent_coord_y[j % 4] - self.elem_rad, \
                        elem_ref_cent_coord_y[j % 4] + self.elem_rad)

                    # Calculate centroid of element by doing weighted sum
                    for j in range(self.elem_size - 2):
                        for k in range(self.elem_size - 2):
                            sum_x += self._image[elem_pix_coord_x[k], elem_pix_coord_y[j]] * elem_pix_coord_x[k]
                            sum_y += self._image[elem_pix_coord_x[k], elem_pix_coord_y[j]] * elem_pix_coord_y[j]
                            sum_pix += self._image[elem_pix_coord_x[k], elem_pix_coord_y[j]]
                     
                    elem_cent_coord_x = sum_x // sum_pix
                    elem_cent_coord_y = sum_y // sum_pix
                    elem_cent_coord = elem_cent_coord_y * self.sensor_width + elem_cent_coord_x