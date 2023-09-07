# TableExtractor
This repository is made for extracting tables from raw photo taken with a phone using Python 3. This library is easy to understand, use and modify to suit your needs.

The objective of our project is to create the most versatile and efficient solution for digitizing a table with data on blood donation from an unpreprocessed photograph of an A4 sheet.

An example of such a photo, on the basis of which the solution will be explained further:
![253721954-6639651d-9f10-4bc0-872a-9f1ceca5d35e](https://github.com/tupperq/TableExtractor/assets/124534158/1714e418-832c-4382-9162-aec221a035d4)

## Preprocessing

Our work began with image pre-processing, namely with recognition of a table with data on donations on an A4 sheet and its further preparation.

Not every pre-trained model is suitable for working with unpreprocessed images, with a slope,
imperfect brightness and a little wrinkled, which is why this stage represents a rather complex solution from me.

### Table recognition

During testing, we found the most suitable model, namely __TahaDouaji/detr-doc-table-detection__ , which is built on dert-resnet-50 and therefore copes well with photographs of our task.

```ruby
image = Image.open(file_path).convert("RGB")

processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
```

As a result, we get __coordinates of the table__, relative to the upper left corner of the photo in the format [__x, y__ - upper left corner of the table, __x, y__ - lower right corner of the table]

Sometimes just a couple of pixels are missing for the table to be clearly recognized. This happens due to the tilt of the image and wrinkling on it. Therefore, it was decided to make the coordinates more universal.

We stopped cropping the bottom of the table, and subtracted a few pixels from the top coordinates, which helped us get a full-fledged table.

```ruby
orig_size = list(image.size)
box[3] = orig_size[1]
box[1] = box[1] - 75
box[0] = 0
box[2] = orig_size[0]
```
![Без названия](https://github.com/tupperq/TableExtractor/assets/124534158/c5d49ca0-8eb9-478d-b97f-53ac361da9a0)

Next, we were faced with the task of leaving only the table in the photo. For this we used the OpenCV library.

__Next, the result of preprocessing will be shown on the original image for a better understanding of the process__

Initially, the picture was binarized, white and black were swapped to make edge recognition easier, and table and text edges were thickened for easier future recognition as well.

```ruby
np_image = np.asarray(image)
grayscale_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
inverted_image = cv2.bitwise_not(thresholded_image)
dilated_image = cv2.dilate(inverted_image, None, iterations=5)
```
Then we found any shapes on the image (contours, text, etc.).

```ruby
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_all_contours = np_image.copy()
cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
```
![Без названия (1)](https://github.com/tupperq/TableExtractor/assets/124534158/3abcb4e2-8624-4b7a-810e-e7c0d8cb50bf)

Next, we got rid of the text selection, leaving only the outlines of the table cells.

```ruby
rectangular_contours = []
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        rectangular_contours.append(approx)

image_with_only_rectangular_contours = np_image.copy()
cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
```
![Без названия (2)](https://github.com/tupperq/TableExtractor/assets/124534158/65af016c-05f4-4374-bbd0-c2d7c65e854e)

Thanks to this, we were able to find the largest rectangle, which is the outline of the entire table.

```ruby
max_area = 0
contour_with_max_area = None
for contour in rectangular_contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        contour_with_max_area = contour

image_with_contour_with_max_area = np_image.copy()
cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)

```
![Без названия (3)](https://github.com/tupperq/TableExtractor/assets/124534158/24f65618-09e4-408d-898b-573519574792)

At this stage, we have a clearly defined table, now it remains only to remove all unnecessary around it.
To accomplish this task, we found the edges of the resulting contour.

```ruby
def order_points(pts):

        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


contour_with_max_area_ordered = order_points(contour_with_max_area)
image_with_points_plotted = np_image.copy()
for point in contour_with_max_area_ordered:
        point_coordinates = (int(point[0]), int(point[1]))
        image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
cv2_imshow(image_with_points_plotted)
```
![Без названия (4)](https://github.com/tupperq/TableExtractor/assets/124534158/82eab1ab-70c9-4505-a9ed-716b55062c60)

And they rotated the table, cutting off everything superfluous, getting a perfectly processed image.

```ruby
def calculateDistanceBetween2Points(p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

existing_image_width = np_image.shape[1]
existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

distance_between_top_left_and_top_right = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[1])
distance_between_top_left_and_bottom_left = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[3])
aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right
new_image_width = existing_image_width_reduced_by_10_percent
new_image_height = int(new_image_width * aspect_ratio)

pts1 = np.float32(contour_with_max_area_ordered)
pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective_corrected_image = cv2.warpPerspective(dilated_image, matrix, (new_image_width, new_image_height))
perspective_corrected_orig_image = cv2.warpPerspective(np_image, matrix, (new_image_width, new_image_height))
cv2_imshow(perspective_corrected_orig_image)
```

![Без названия (5)](https://github.com/tupperq/TableExtractor/assets/124534158/a39ef9cd-2b30-4f19-88fb-bc177479d054)

## Table outline detection

At this stage, the task was to __recognize all the horizontal and vertical lines__ of the table in order to find their coordinates for a clear distinction between cells, and to remove them for recognition.

__Stage task: get an image without table outlines__

To solve this problem, we found the horizontal lines of the table, then the vertical ones and combined them.

```ruby
hor = np.array([[1,1,1,1,1,1]])
vertical_lines_eroded_image = cv2.erode(perspective_corrected_image, hor, iterations=100)
vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=100)
ver = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
horizontal_lines_eroded_image = cv2.erode(perspective_corrected_image, ver, iterations=100)
horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=100)
combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=7)
```
![Без названия (6)](https://github.com/tupperq/TableExtractor/assets/124534158/512c164e-d483-47d6-a94b-2895c9c070ef)

Then we __removed the resulting contours__ in two steps:

```ruby
image_without_lines = cv2.subtract(perspective_corrected_image, combined_image_dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=5)
image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=5)
```
![Без названия (7)](https://github.com/tupperq/TableExtractor/assets/124534158/0de4142d-b620-4b6e-89cd-ccb1972c605d)

## Text recognition

__We decided to do this task in the following sequence:__
- Recognize exactly where the text is located and get the coordinates of these places.
- Get the sequence of rows and columns.
- Cut each text element separately and recognize it.
- Assemble the table back based on the sequence of rows and columns.

To begin with, we greatly increased all the elements in the image in order to find those frames where any text is located.

```ruby
kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1]
        ])
dilated_image = cv2.dilate(image_without_lines_noise_removed, kernel_to_remove_gaps_between_words, iterations=5)
simple_kernel = np.ones((5,5), np.uint8)
dilated_image = cv2.dilate(image_without_lines_noise_removed, simple_kernel, iterations=5)
```
![Без названия (8)](https://github.com/tupperq/TableExtractor/assets/124534158/93d232b4-1853-4059-b9a0-d6b61ead879d)

Next, we found the outlines of the white shapes and transferred them to the original image. The outlines were converted to a rectangular format, which helped to recognize the text much better.

```ruby
result = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = result[0]
image_with_contours_drawn = perspective_corrected_orig_image.copy()
cv2.drawContours(image_with_contours_drawn, contours, -1, (0, 255, 0), 3)

approximated_contours = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, 3, True)
    approximated_contours.append(approx)

image_with_contours = perspective_corrected_orig_image.copy()
cv2.drawContours(image_with_contours, approximated_contours, -1, (0, 255, 0), 5)

bounding_boxes = []
image_with_all_bounding_boxes = perspective_corrected_orig_image.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))
    image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
```

![Без названия (9)](https://github.com/tupperq/TableExtractor/assets/124534158/4f6eb738-5e60-4455-ae13-f2bc4a2f22e1)

Next, we got a representation of the location of the cells relative to each other:


```ruby
def get_mean_height_of_bounding_boxes():
    heights = []
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(h)
    return np.mean(heights)
bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
mean_height = get_mean_height_of_bounding_boxes()

rows = []
half_of_mean_height = mean_height / 2
current_row = [ bounding_boxes[0] ]
for bounding_box in bounding_boxes[1:]:
    current_bounding_box_y = bounding_box[1]
    previous_bounding_box_y = current_row[-1][1]
    distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
    if distance_between_bounding_boxes <= half_of_mean_height:
        current_row.append(bounding_box)
    else:
        rows.append(current_row)
        current_row = [ bounding_box ]
rows.append(current_row)
for row in rows:
            row.sort(key=lambda x: x[0])
```

Next step is the most interesting. According to the obtained coordinates, we crop each cell and use __Tesseract-OCR__ to recognize the text in it. Then we arrange everything in the matrix in the same order as we took it.

Tesseract:
```ruby
 - -l rus --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="йцукенгшщзхъфывапролджэячсмитьбю/ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ0123456789().calmg* 
```

```ruby
table = []
current_row = []
image_number = 0
for row in rows:
    for bounding_box in row:
        x, y, w, h = bounding_box
        cropped_image = perspective_corrected_orig_image[y:y+h, x:x+w]
        image_slice_path = "./ocr_slices/img_" + str(image_number) + ".jpg"
        cv2.imwrite(image_slice_path, cropped_image)
        results_from_ocr = get_result_from_tersseract(image_slice_path)
        current_row.append(results_from_ocr)
        image_number += 1
    table.append(current_row)
    current_row = []
```

At the output, we get an unedited table with data:


![Screenshot_592](https://github.com/tupperq/TableExtractor/assets/124534158/31ef51c9-5083-4ea6-b217-1376fd0cbd63)

## Format recognition result

The formatting of our result began with filtering the data in the table. Tesseract errors and all elements that did not suit us were removed.

```ruby
def delete_redundant_elements(table, iter=5):

    filtered_table = table.copy()

    for _ in range(iter):

        for row in filtered_table:

            for item in row:
                if (len(item) < 6) or (len(item) > 12):
                    row.remove(item)

    return filtered_table


def get_max_row_lenght(table):

    max = 0

    for row in table:
        if len(row) > max:
            max = len(row)

    return max


def delete_redundant_rows(table, iter=3):

    filtered_table = table.copy()
    max = get_max_row_lenght(table)

    for _ in range(iter):
        for row in filtered_table:
            if (len(row) <= max / 3) or (len(row) <= 2):
                filtered_table.remove(row)

    return filtered_table


def split_don_type(table):

    filtered_table = []

    for row in table:
        filtered_table.append(
            [splitted_item for item in row for splitted_item in item.split()]
            )

    return filtered_table


def split_long_row(table):

    updated_table = []

    for row in table:
        if len(row) > 10:
            half_length = len(row) // 2
            updated_table.append(row[:half_length])
            updated_table.append(row[half_length:])
        else:
            updated_table.append(row)

    return updated_table


def change_values(value: str, values: dict) -> str:
    if value in values.keys():
        return values[value]
    else:
        return value


def raw_table_filter(raw_pred):
    filtered_table = delete_redundant_elements(raw_pred)
    filtered_table = delete_redundant_rows(filtered_table)
    filtered_table = split_don_type(filtered_table)

    return filtered_table
```

Next, it was necessary to arrange the received data in the correct order __without confusing the cells__, since not all elements can be recognized

```ruby
max_len = 0
for row in filtered_table:
    if len(row) > max_len:
        max_len = len(row)
row_len = 3
new_table = []
for i in range(len(filtered_table) * int(max_len / 3)):
    new_row = [0 for _ in range(row_len)]
    new_table.append(new_row)
counter = 0
row_counter = 0
if max_len == 8:
    max_len += 1

for i in range(len(filtered_table)):

    if max_len == 6:
        pass
    elif max_len == 9 and new_table[row_counter][2] == 0 and new_table[row_counter].count(0) < 3:
        row_counter += 0
    elif max_len == 9 and new_table[row_counter].count(0) == 3:
        if row_counter % 3 == 1:
            row_counter += 2
        elif row_counter % 3 == 2:
            row_counter += 1
    for j in range(len(filtered_table[i])):
        counter = 0
        try:
            datetime_object = pd.to_datetime(filtered_table[i][j].strip('.'), format='%d.%m.%Y')
            try:
                if new_table[row_counter - 1][2] == 0 and row_counter != 0:
                    row_counter += 1
            except:
                pass
            if new_table[row_counter][counter] != 0:
                row_counter += 1
            new_table[row_counter][counter] = filtered_table[i][j].strip('.')
            continue

        except:
            counter += 1

        if filtered_table[i][j] in don_type.keys():
            if new_table[row_counter][counter] != 0:
                row_counter += 1
            new_table[row_counter][counter] = change_values(filtered_table[i][j])
            continue
        else:
            counter += 1

        if filtered_table[i][j] in pay_type.keys():
            new_table[row_counter][counter] = change_values(filtered_table[i][j])
            row_counter += 1
            continue
        else:
            counter += 1

new_table = pd.DataFrame(new_table, columns =  ['Дата донации', 'Класс крови', 'Тип донации'])
```

As a result, we get the following table:

![photo_2023-07-17_21-07-11](https://github.com/tupperq/TableExtractor/assets/124534158/36b6c97d-b05d-4d83-9811-3f41dc5135e9)

## Accuracy

To calculate the __accuracy__, we decided to evaluate both the location and the correct recognition of each cell.

```ruby
def accuracy_score(table_pred, table_true):
    print(table_pred.shape, table_true.shape)
    if table_pred.shape == table_true.shape:

        rows = int(table_pred.shape[0])
        cols = int(table_pred.shape[1])
        total = rows * cols
        correct = 0

        for row in range(rows):
            for col in range(cols):
                if table_pred.iloc[row, col] == table_true.iloc[row, col]:
                    correct += 1
                else:
                    continue

        return correct / total

    else:
        print('Shapes of table_pred and table_true does not match!')
```

__Average Accuracy__ = 0.86 = 87%
