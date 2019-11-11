def get_SB_geometry(self):
        """
        Calculates search block geometry from given number of spots across diameter
        """
        # Clear search block layer
        self.SB_layer_2D = np.zeros([self.sensor_width, self.sensor_height], dtype='uint8')
        self.SB_layer_1D = np.ravel(self.SB_layer_2D)

        # Initialise list of 1D coords of reference centroids within pupil diameter
        self.act_ref_cent_coord = []

        # Get pupil diameter for given number of spots
        spots_along_diag = self.SB_across_width

        try:
            self.spots_across_diam > 2 and self.spots_across_diam < spots_along_diag
        except ValueError as ex:
            ex_type = sys.exc_info()
            logger.error('Number of spots across diameter is out of bounds.')
            print('Exception type: %s ' % ex_type.__name__)

        if (self.spots_across_diam < 2 or self.spots_across_diam > spots_along_diag):
            print('Number of spots across diameter is out of bounds.')

        for i in range(spots_along_diag // 2):
            
            if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
                (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

                pupil_rad_temp_max = np.sqrt(((self.sensor_width // 2 - \
                    (self.ref_cent_x[i] + 1) + self.SB_rad) * self.pixel_size) ** 2 + ((self.sensor_height // 2 - \
                    self.ref_cent_y[i] + self.SB_rad) * self.pixel_size) ** 2)
                pupil_rad_temp_min = np.sqrt(((self.sensor_width // 2 - \
                    (self.ref_cent_x[i + 1] + 1) + self.SB_rad) * self.pixel_size) ** 2 + ((self.sensor_height // 2 - \
                    self.ref_cent_y[i + 1] + self.SB_rad) * self.pixel_size) ** 2)

            else:

                pupil_rad_temp_max = np.sqrt(((self.sensor_width // 2 - \
                    (self.ref_cent_x[i] + 1)) * self.pixel_size) ** 2 + ((self.sensor_height // 2 - \
                    self.ref_cent_y[i]) * self.pixel_size) ** 2)
                pupil_rad_temp_min = np.sqrt(((self.sensor_width // 2 - \
                    (self.ref_cent_x[i + 1] + 1)) * self.pixel_size) ** 2 + ((self.sensor_height // 2 - \
                    self.ref_cent_y[i + 1]) * self.pixel_size) ** 2)

            if self.spots_across_diam % 2 == 1:
                pixel_edge = (self.SB_diam * (self.spots_across_diam // 2) + self.SB_rad) * self.pixel_size
            else:
                pixel_edge = self.SB_diam * (self.spots_across_diam // 2) * self.pixel_size
            
            if pupil_rad_temp_max > pixel_edge and pupil_rad_temp_min < pixel_edge:
                self.pupil_rad = pupil_rad_temp_max
                break

        # Get reference centroids within pupil diameter
        if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
            (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

            for j in self.ref_cent_y:
                for i in self.ref_cent_x:
                    if ((np.sqrt(((abs((i + 1 - self.sensor_width // 2)) + self.SB_rad) * self.pixel_size) ** 2 + \
                        ((abs((j + 1 - self.sensor_height // 2)) + self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_rad):
                        self.act_ref_cent_coord.append(j * self.sensor_width + i)

        else:

            for j in self.ref_cent_y:
                for i in self.ref_cent_x:
                    if ((np.sqrt(((abs((i + 1 - (self.sensor_width // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2 + \
                        ((abs((j + 1 - (self.sensor_height // 2 - self.SB_rad))) + self.SB_rad) * self.pixel_size) ** 2)) < self.pupil_rad):
                        self.act_ref_cent_coord.append(j * self.sensor_width + i)

        # Set actual search block reference centroids
        self.act_ref_cent_coord = np.array(self.act_ref_cent_coord)
        self.act_ref_cent_num = len(self.act_ref_cent_coord)
        self.SB_layer_1D[self.act_ref_cent_coord] = self.outline_int
        self.SB_layer_2D = np.reshape(self.SB_layer_1D, (self.sensor_height, self.sensor_width))

        # Get 1D coord offset of each actual search block
        act_ref_cent_offset_top_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width - self.SB_rad
        act_ref_cent_offset_bottom_coord = self.act_ref_cent_coord + self.SB_rad * self.sensor_width - self.SB_rad
        act_ref_cent_offset_right_coord = self.act_ref_cent_coord - self.SB_rad * self.sensor_width + self.SB_rad

        # Get 2D coord offset of each actual search block
        act_ref_cent_offset_top_y = act_ref_cent_offset_top_coord // self.sensor_width
        act_ref_cent_offset_top_x = act_ref_cent_offset_top_coord % self.sensor_width
        act_ref_cent_offset_bottom_y = act_ref_cent_offset_bottom_coord // self.sensor_width
        act_ref_cent_offset_bottom_x = act_ref_cent_offset_bottom_coord % self.sensor_width
        act_ref_cent_offset_right_y = act_ref_cent_offset_right_coord // self.sensor_width
        act_ref_cent_offset_right_x = act_ref_cent_offset_right_coord % self.sensor_width

        # Get parameters for outlining actual search blocks
        act_ref_row_top_outline, row_top_indices, row_top_counts =\
            np.unique(act_ref_cent_offset_top_y, return_index = True, return_counts = True)
        act_ref_row_bottom_outline, row_bottom_indices, row_bottom_counts =\
            np.unique(act_ref_cent_offset_bottom_y, return_index = True, return_counts = True)
        act_ref_column_left_outline, column_left_indices, column_left_counts =\
            np.unique(act_ref_cent_offset_top_x, return_index = True, return_counts = True)
        act_ref_column_right_outline, column_right_indices, column_right_counts =\
            np.unique(act_ref_cent_offset_right_x, return_index = True, return_counts = True)

        # Get number of rows and columns for outlining
        rows = len(act_ref_row_top_outline)
        columns = len(act_ref_column_left_outline)

        # Outline rows of actual search blocks
        for i in range(rows // 2 + 1):
            self.SB_layer_2D[act_ref_row_top_outline[i], act_ref_cent_offset_top_x[row_top_indices[i]] :\
                act_ref_cent_offset_top_x[row_top_indices[i + 1] - 1] + self.SB_diam] = self.outline_int

        for i in range(rows // 2, rows):
            self.SB_layer_2D[act_ref_row_bottom_outline[i], act_ref_cent_offset_bottom_x[row_bottom_indices[i]] :\
                act_ref_cent_offset_bottom_x[row_bottom_indices[i] + row_bottom_counts[i]- 1] + self.SB_diam] = self.outline_int    

        # Outline columns of actual search blocks
        self.index_count = 0
        
        for i in range(columns // 2 + 1):

            if i == 0:
                self.SB_layer_2D[act_ref_cent_offset_top_x[self.index_count] : act_ref_cent_offset_top_x[self.index_count + \
                    column_left_counts[i] - 1] + self.SB_diam, act_ref_column_left_outline[i]] = self.outline_int
            else:
                self.index_count += column_left_counts[i - 1] 
                self.SB_layer_2D[act_ref_cent_offset_top_x[self.index_count] : act_ref_cent_offset_top_x[self.index_count + \
                    column_left_counts[i] - 1] + self.SB_diam, act_ref_column_left_outline[i]] = self.outline_int

        for i in range(columns // 2, columns):

            if i == columns // 2:
                self.SB_layer_2D[act_ref_cent_offset_right_x[self.index_count] - self.SB_diam : act_ref_cent_offset_right_x[self.index_count \
                    + column_right_counts[i] - 1], act_ref_column_right_outline[i]] = self.outline_int
            else:
                self.index_count += column_left_counts[i - 1] 
                self.SB_layer_2D[act_ref_cent_offset_right_x[self.index_count] - self.SB_diam : act_ref_cent_offset_right_x[self.index_count \
                    + column_right_counts[i] - 1], act_ref_column_right_outline[i]] = self.outline_int  

        # Draw pupil circle on search block layer
        plot_point_num = int(self.pupil_rad * 2 // self.pixel_size * 10)

        theta = np.linspace(0, 2 * math.pi, plot_point_num)
        rho = self.pupil_rad // self.pixel_size

        if (self.spots_across_diam % 2 == 0 and self.sensor_width % 2 == 0) or \
            (self.spots_across_diam % 2 == 1 and self.sensor_width % 2 == 1):

            x = (rho * np.cos(theta)).astype(int) + self.sensor_width // 2
            y = (rho * np.sin(theta)).astype(int) + self.sensor_width // 2

        else:

            x = (rho * np.cos(theta)).astype(int) + self.sensor_width // 2 - self.SB_rad
            y = (rho * np.sin(theta)).astype(int) + self.sensor_width // 2 - self.SB_rad

        self.SB_layer_2D[x, y] = self.outline_int

        # Display actual search blocks and reference centroids
        img = PIL.Image.fromarray(self.SB_layer_2D, 'L')
        img.show()

        # Get actual search block coordinates
        self.act_SB_coords = np.nonzero(np.ravel(self.SB_layer_2D))