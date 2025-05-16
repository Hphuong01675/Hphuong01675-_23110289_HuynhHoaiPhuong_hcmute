# AI_PersonalProject

## 1. Mục tiêu

- Đồ án là tập hợp các bài tập hàng tuần trong suốt khóa học nhằm tổng hợp kiến thức và áp dụng để mô phỏng các thuật toán tìm kiếm đã học vào bài toán 8-Puzzle.
- Qua đồ án, sinh viên được củng cố kiến thức trên lớp thông qua việc mô phỏng lại các thuật toán đã học. Từ đó, sinh viên có thể nắm chắc kiến thức hơn về các thuật toán.
- Ngoài ra, sinh viên được tiếp cận với môi trường trên Github và có được một sản phẩm nhỏ trên kho Github của bản thân.

## 2. Nội dung

#### 2.1. Các thuật toán Tìm kiếm không có thông tin (Uninformed Search Algorithms)

##### Thành phần chính của bài toán tìm kiếm
- **Không gian tìm kiếm (State-Space)**: tập các giải pháp khả thi.
- **Trạng thái ban đầu (Initial State)**: trạng thái mà tác nhân bắt đầu tìm kiếm.
- **Trạng thái đích (Goal State)**: điều kiện để thuật toán dừng, cũng là trạng thái mong muốn.
- **Tập hành động (Actions)**: tập các hành động mà tác tử có thể thực hiện để chuyển từ trạng thái này sang trạng thái khác.
- **Mô hình chuyển tiếp (Transition Model)**: chức năng của từng hành động, kết quả của hành động tại một trạng thái cụ thể.
- **Chi phí đường đi (Path Cost)**: hàm tính chi phí cho mỗi đường đi.
- **Giải pháp (Solution)**: chuỗi các hành động dẫn từ trạng thái ban đầu đến trạng thái đích.
- **Giải pháp tối ưu (Best Solution)**: giải pháp có chi phí thấp nhất.

##### Breadth-First Search (BFS):
- Duyệt theo bề rộng, mở rộng tất cả các nút cùng mức trước khi chuyển sang mức sâu hơn.
- Open List: Danh sách liên kết hoặc hàng đợi.
- Độ phức tạp thời gian: $1 + b + b^2 + ... + b^d = O(b^d)$
- Độ phức tạp không gian: $O(b^d)$

![BFS](/DoAnCaNhan_gif/BFS.gif)

##### Depth-First Search (DFS):
- Ưu tiên mở rộng theo chiều sâu, đi theo một nhánh cho đến khi không thể đi tiếp
- Open List: Ngăn xếp.
- Độ phức tạp thời gian: $T(n)= 1+ n^2 + n^3 +...+ n^m =O(n^m)$
- Độ phức tạp không gian: $O(bm)$

![DFS](/DoAnCaNhan_gif/DFS.gif)

##### Uniform Cost Search (UCS):
- Mở rộng nút có chi phí đường dẫn tối thiểu.
- Open List: Hàng đợi ưu tiên.
- Độ phức tạp thời gian: $O(b^{1 + [C*/ε]})$
- Độ phức tạp không gian: $O(b^{1 + [C*/ε]})$

![UCS](/DoAnCaNhan_gif/UCS.gif)

##### Iterative Deepening Search (IDS):
- Kết hợp ưu điểm của BFS và DFS. Tìm theo chiều sâu nhưng giới hạn độ sâu, tăng dần qua mỗi vòng.
- Open List: Ngăn xếp.
- Độ phức tạp tính toán: $O(b^d) = (d+1)b^0 + db^1 + (d-1)b^2 + … + b^d$
- Độ phức tạp thời gian: $O(b^d)$
- Độ phức tạp không gian: $O(bd)$ 

![IDS](/DoAnCaNhan_gif/IDS.gif)

##### So sánh các thuật toán trong nhóm
![Comparison](/DoAnCaNhan_gif/Uninformed_Search.jpg)
![Comparison](/DoAnCaNhan_gif/Uninformed_Search_2.jpg)

#### 2.2. Các thuật toán Tìm kiếm có thông tin (Informed Search Algorithms)

##### Thành phần chính của bài toán tìm kiếm
- Hàm đánh giá f(N) cho mỗi nút của cây tìm kiếm để đánh giá mức độ “phù hợp” của nút đó.
- Hàm ước lượng chi phí (heuristic) h(N) không âm (nếu bằng không N là node đích).

##### Greedy Search:
- Sử dụng hàm heuristic h(N) để ước lượng khoảng cách (chi phí) từ các nodes tới node đích và Solution sẽ ưu tiên node có giá trị hàm h(N) nhỏ nhất.
- Open List: hàng đợi ưu tiên.
- Độ phức tạp về thời gian: trong trường hợp xấu nhất là $O(b^m)$
    - b là số nhánh của cây/đồ thị
    - m là độ sâu của cây/đồ thị khi tìm ra Solution
- Độ phức tạp về không gian: trong trường hợp xấu nhất là $O(b^m)$

![Greedy](/DoAnCaNhan_gif/Greedy.gif)

##### A* Search:
- Sử dụng hàm đánh giá f(N) = g(N) + h(N)
    - g(N) = chi phí từ node gốc cho đến node hiện tại N.
    - h(N) = chi phí ước lượng từ nút hiện tại n tới đích.
    - f(N) = chi phí tổng thể ước lượng của đường đi qua nút hiện tại N đến đích.
- Open List: hàng đợi ưu tiên
- Độ phức tạp tính toán và yêu cầu bộ nhớ của A* đều là $O(b^m)$

![Astar](/DoAnCaNhan_gif/A_star.gif)

##### IDA* Search:
- Mở rộng từ thuật toán A*, sử dụng các khái niệm về DFS lặp lại - làm sâu dựa trên các ngưỡng đặt trước.
- Open List: không lưu toàn bộ danh sách, mở rộng dần theo chiều sâu đến khi đạt ngưỡng.
- Độ phức tạp thời gian: $O(b^d)$
- Độ phức tạp không gian: $O(lb)$
    - l: chiều dài của path được tạo dài nhất
    - b: số nhánh
    - d: là độ sâu

![IDAstar](/DoAnCaNhan_gif/IDA_star.gif)

##### So sánh các thuật toán trong nhóm
![Comparison](/DoAnCaNhan_gif/Informed_Search.jpg)

#### 2.3. Các thuật toán Tìm kiếm cục bộ (Local Search Algorithms)

##### Thành phần chính của bài toán tìm kiếm
- Trạng thái hiện tại
- Trạng thái lân cận (Hàng xóm)
- Hàm đánh giá

##### Leo đồi đơn giản (Simple hill Climbing) 
Chỉ kiểm tra từng trạng thái lận cận của nó và nếu nó tìm thấy trạng thái tốt hơn trạng thái hiện tại thì di chuyển.

- BƯỚC 1: Chọn trạng thái hiện tại.
- BƯỚC 2: Tạo một hàng xóm của trạng thái hiện tại.
- BƯỚC 3: Đánh giá hàm mục tiêu tại hàng xóm được đề xuất.
- BƯỚC 4: Nếu giá trị của hàm mục tiêu tại hàng xóm được đề xuất tốt hơn tại trạng thái hiện tại, thì trạng thái hàng xóm được đề xuất sẽ trở thành trạng thái hiện tại.
- BƯỚC 5: Lặp lại BƯỚC 2-BƯỚC 4 trong n lần lặp hoặc cho đến khi giá trị của mục tiêu tại trạng thái hiện tại cao hơn tất cả các hàng xóm.
- BƯỚC 6: Trả về trạng thái hiện tại và giá trị hàm mục tiêu của nó.

![SimpleHC](/DoAnCaNhan_gif/SH.gif)

##### Leo đồi dốc nhất (Steepest-Ascent hill-climbing)
Thuật toán này kiểm tra tất cả các nút lân cận của trạng thái hiện tại và chọn một nút lân cận gần nhất với trạng thái mục tiêu.

- BƯỚC 1: Chọn trạng thái hiện tại.
- BƯỚC 2: Tạo tất cả các trạng thái lân cận của trạng thái hiện tại.
- BƯỚC 3: Đánh giá hàm mục tiêu tại tất cả các lân cận.
- BƯỚC 4: Nếu hàm mục tiêu tại trạng thái hiện tại có giá trị cao hơn tất cả các trạng thái lân cận, thì trạng thái hiện tại đã là giá trị tối đa: KẾT THÚC TÌM KIẾM và nhảy đến BƯỚC 6. Nếu không, lân cận có cải thiện cao nhất của hàm mục tiêu sẽ trở thành trạng thái hiện tại.
- BƯỚC 5: Lặp lại BƯỚC 2-BƯỚC 4 trong n lần lặp.
- BƯỚC 6: Trả về trạng thái hiện tại và giá trị hàm mục tiêu của nó.

![SteepestHC](/DoAnCaNhan_gif/SteepestH.gif)

##### Leo đồi ngẫu nhiên (Stochastic hill Climbing)
Lựa chọn ngẫu nhiên một hàng xóm. Nếu hàng xóm đó tốt hơn trạng thái hiện tại, hàng xóm đó sẽ được chọn làm trạng thái hiện tại và thuật toán lặp lại. Ngược lại, nếu hàng xóm được chọn không tốt hơn, thuật toán sẽ chọn ngẫu nhiên một hàng xóm khác và so sánh. Thuật toán kết thúc và trả lại trạng thái hiện tại khi đã hết “kiên nhẫn” (vượt ngưỡng).

- BƯỚC 1: Chọn trạng thái hiện tại.
- BƯỚC 2: Tạo tất cả các trạng thái lân cận của trạng thái hiện tại.
- BƯỚC 3: Đánh giá hàm mục tiêu tại tất cả các lân cận.
- BƯỚC 4: Nếu hàm mục tiêu tại trạng thái hiện tại có giá trị cao hơn tất cả các trạng thái lân cận, thì trạng thái hiện tại đã là giá trị tối đa: KẾT THÚC TÌM KIẾM và nhảy đến BƯỚC 6. Nếu
không, lân cận có cải thiện cao nhất của hàm mục tiêu sẽ trở
thành trạng thái hiện tại.
- BƯỚC 5: Lặp lại BƯỚC 2-BƯỚC 4 trong n lần lặp.
- BƯỚC 6: Trả về trạng thái hiện tại và giá trị hàm mục tiêu của nó.

![StochasticHC](/DoAnCaNhan_gif/StoH.gif)

##### Simulated Annealing
Cho phép thoát khỏi đỉnh cục bộ bằng cách chấp nhận trạng thái kém hơn theo xác suất, giảm dần theo "nhiệt độ".

![SimulatedAnnealing](/DoAnCaNhan_gif/Simulated_Annealing.gif)

##### Beam Search
Giữ lại k trạng thái tốt nhất tại mỗi bước (k-beam). Giống BFS nhưng giới hạn độ rộng.

![BeamSearch](/DoAnCaNhan_gif/Beam_Search.gif)

##### Genetic Algorithms
Lấy cảm hứng từ di truyền học: khởi tạo quần thể, chọn lọc, lai ghép, đột biến để tạo thế hệ tiếp theo.

![GeneticAlgo](/DoAnCaNhan_gif/GA.gif)

##### So sánh các thuật toán trong nhóm
![Comparison](/DoAnCaNhan_gif/Local_Search.jpg)
![Comparison](/DoAnCaNhan_gif/Local_search_2.jpg)
- **Chú thích**:
    - SH: Stochastic Hill Climbing
    - SA: Simulated Annealing

#### 2.4. Các thuật toán Tìm kiếm trong môi trường phức tạp (Complex Environment)

##### Phân loại môi trường
|Loại môi trường|Không xác định|Không quan sát|Quan sát một phần
|:----|:---|:---|:------|
|Deterministic|✅|❌|❌|
|No Observation|✅|✅|❌|
|Partially Observable|✅|❌|✅|

##### Search in Nondeterministic Environment
- Hành động có thể có nhiều kết quả khác nhau.
- Không biết trước được trạng thái tiếp theo là gì.
- Cấu trúc tìm kiếm: AND-OR TREE.
- Giải pháp: dùng AND-OR Search.

![Nondeterministic](/DoAnCaNhan_gif/AND_OR.gif)

##### Search with No Observation
- Không có thông tin gì về trạng thái hiện tại.
- Chiến lược:
    - Bắt đầu từ tập hợp các trạng thái niềm tin (belief state).
    - Thử các hành động áp dụng lên toàn bộ các trạng thái đó.
    - Lặp lại cho đến khi tất cả trạng thái đều đạt trạng thái đích.
- Giải pháp: xây dựng Belief State Search (duy trì tập hợp trạng thái niềm tin)

![NoObservation](/DoAnCaNhan_gif/Belief.gif)

##### Partially Observable Search
- Tác nhân không thấy toàn bộ môi trường, nhưng có thể cảm nhận một phần (ví dụ qua cảm biến).
- Giải pháp: dùng belief update sau mỗi hành động và quan sát.

$\to$ Nhóm thuật toán này trong đồ án hiện đang phát triển.
##### So sánh các thuật toán trong nhóm
![Comparison](/DoAnCaNhan_gif/Complex_Environment.jpg)
![Comparison](/DoAnCaNhan_gif/Complex_Environment_2.jpg)

#### 2.5. Các thuật toán Tìm kiếm thỏa mãn ràng buộc (Constraint Satisfaction Problem)

##### Thành phần chính của bài toán tìm kiếm
- Tập biến (Variables)
- Miền giá trị (Doamins)
- Ràng buộc (Constraints)

$\to$ Bài toán ràng buộc (CSP) là dạng bài toán trong đó mỗi biến phải được gán giá trị sao cho thỏa mãn tập các ràng buộc đã cho.

##### Backtracking
- Thử gán giá trị cho biến theo thứ tự, nếu ràng buộc bị vi phạm thì quay lui lại và thử giá trị khác.
- Open List: ngăn xếp
- Độ phức tạp thời gian: $O(d^n)$
- Độ phức tạp không gian: $O(n)$
    - d: số miền
    - n: số biến
 
  
![Backtracking](/DoAnCaNhan_gif/Backtracking.gif)
##### Forward Checking
- Bản cải tiến của Backtracking. Sau khi gán giá trị cho một biến, loại bỏ các giá trị bất hợp lệ ra khỏi miền của các biến liên quan.
- Open List: danh sách miền cập nhật cho từng biến.
- Độ phức tạp thời gian: $O(nd^2)$
- Độ phức tạp không gian: $O(nd)$

  
![Backtracking with forward checking](/DoAnCaNhan_gif/Forward_checking.gif)
##### Min-Conflict
- Bắt đầu từ một gán ngẫu nhiên, sau đó lặp lại: 
    - Chọn biến vi phạm ràng buộc.
    - Đổi giá trị sao cho số lượng xung đột nhỏ nhất.

- Độ phức tạp thời gian: $O(n)$
- Độ phức tạp không gian: $O(n)$


![Min-Conflict](/DoAnCaNhan_gif/Min_conflict.gif)

##### So sánh các thuật toán trong nhóm
![Comparison](/DoAnCaNhan_gif/CPSs.jpg)
![Comparison](/DoAnCaNhan_gif/CSPs_1.jpg)

#### 2.6. Các thuật toán Tìm kiếm học tăng cường (Reinforcement Learning)

##### Thành phần chính của bài toán tìm kiếm
- Agent: thuật toán hoặc hệ thống ra quyết định trong RL.
- Environment: bối cảnh mà tác tử hoạt động, bao gồm các quy tắc, giới hạn và các biến số.
- State: bối cảnh của môi trường tại một thời điểm nhất định.
- Action: các bước mà tác tử có thể thực hiện để thay đổi trạng thái của môi trường. 
- Reward: giá trị phản hồi (có thể dương, âm hoặc bằng không) mà tác tử nhận được sau mỗi hành động, phản ánh mức độ hiệu quả của hành động đó.
- Cumulative Reward: tổng phần thưởng mà tác tử nhận được trong quá trình hoạt động.
- Episode: loạt các tương tác giữa agent và environment từ lúc bắt đầu đến kết thúc.
- Policy: chiến lược mà tác nhân chọn để áp dụng cho hành động tiếp theo dựa trên trạng thái hiện tại.

##### Q-Learning 
(Do sự phức tạp của nhóm thuật toán RL nên trong đồ án này chỉ khai thác và áp dụng cơ bản thuật toán Q-Learning)

- Học bảng Q[s, a] để tìm chính sách tối ưu.
    - Off-policy: Học từ hành động không nhất thiết là hành động đã chọn.
    - Luôn chọn giá trị Q lớn nhất để cập nhật.
- Công thức cập nhật (phương trình Bellman): 
$Q(s,a) ← Q(s,a) + \alpha[r+ \gamma maxQ(s', a') − Q(s,a)]$

$\to$ Đây là thuật toán không phù hợp để giải bài toán 8-Puzzle.

![Q_Learning](/DoAnCaNhan_gif/Q_Learning.gif)

##### SARSA
- Học bảng Q tương tự Q-Learning nhưng theo chính sách hiện tại.
    - On-policy: Cập nhật theo hành động thực sự đã thực hiện.
    - Cẩn trọng hơn, ít mạo hiểm hơn Q-Learning.
- Công thức cập nhật (phương trình Bellman):
$Q(s,a) ← Q(s,a) + \alpha[r+ \gamma Q(s', a') − Q(s,a)]$

##### Deep Q-Network
- Thay bảng Q bằng mạng nơ-ron để dự đoán Q(s, a).
    - Replay buffer: lưu kinh nghiệm để học lại.
    - Target network: giữ mạng Q mục tiêu ổn định.
- Công thức cập nhật:
$loss = (r + \gamma * max(Q_target(s'), a') - Q(s, a))^2$

# Hướng dẫn sử dụng 8-Puzzle Solver
## Bắt đầu
1. Khởi động ứng dụng bằng cách chạy file Python đã cung cấp.
2. Cửa sổ chính sẽ mở ra, hiển thị bảng `puzzle_board`, các điều khiển (`controls`), trạng thái niềm tin (`belief_states`), thống kê (`statistics`), đường đi giải pháp (`solution_path`), và phần đánh giá thuật toán (`algorithm_evaluation`).

## Sử dụng ứng dụng
### 1. Chọn thuật toán
- Tìm menu thả xuống `algorithm` trong phần `controls`.
- Nhấn vào menu để chọn một thuật toán từ danh sách (ví dụ: `BFS`, `DFS`, `A*`, `Hill Climbing`, v.v.).
- Trạng thái ban đầu của `puzzle_board` sẽ được điều chỉnh dựa trên yêu cầu của thuật toán đã chọn.

### 2. Giải puzzle
- Nhấn nút `solve_button` để bắt đầu thuật toán đã chọn.
- Ứng dụng sẽ hiển thị quá trình giải trong phần `solution_path`.
- Đối với các thuật toán như `Belief State Search` hoặc `Search with Partial Observation`, các `belief_states` sẽ được hiển thị trong phần tương ứng.

### 3. Xem kết quả
- Sau khi giải xong, đường đi giải pháp sẽ được hiển thị từng bước trong phần `solution_path`.
- Các thông số thống kê như `running_time`, `steps`, `nodes_expanded`, `max_fringe_size`, và `states_visited` sẽ được cập nhật trong phần `statistics`.
- Các chỉ số đánh giá sẽ được thêm vào bảng `algorithm_evaluation`.

### 4. Xem hoạt ảnh giải pháp
- Bảng `puzzle_board` sẽ tự động hiển thị hoạt ảnh của đường đi giải pháp sau khi giải xong.
- Điều chỉnh tốc độ hoạt ảnh bằng cách thay đổi giá trị `animation_delay` trong mã (mặc định thay đổi theo thuật toán).

### 5. Đặt lại puzzle
- Nhấn nút `reset_button` để trở về trạng thái ban đầu và xóa tất cả kết quả.

### 6. Vẽ cây không gian trạng thái
- Nhấn nút `drawtree_button` để mở một cửa sổ mới hiển thị cây không gian trạng thái.
- Sử dụng bánh xe chuột để phóng to/thu nhỏ và nhấn-giữ-kéo để di chuyển cây.

### 7. Đánh giá thuật toán
- Sử dụng nút `compare_button` để xem so sánh các chỉ số hiệu suất.
- Nhấn `clear_eval_button` để làm mới bảng `algorithm_evaluation`.
- Nhấn `save_eval_button` để lưu dữ liệu vào file CSV.
- Nhấn `chart_button` để xem biểu đồ cột về thời gian chạy của các thuật toán.

## Khắc phục sự cố
- Nếu xảy ra lỗi (ví dụ: trạng thái ban đầu không thể giải được), một thông báo lỗi sẽ hiện lên.
- Hủy thuật toán đang chạy bằng cách sử dụng `reset_button` hoặc chọn thuật toán mới.
- Đảm bảo tất cả các thư viện cần thiết (ví dụ: `tkinter`, `matplotlib`) đã được cài đặt.

## Mẹo
- Thử nghiệm với các thuật toán khác nhau để so sánh hiệu quả của chúng.
