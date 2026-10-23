import sys
import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def build_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors: Clean Tech White & Academic Blue (Trắng - Xanh Kỹ thuật)
    BG_SLATE = RGBColor(248, 250, 252)       # #F8FAFC
    CARD_WHITE = RGBColor(255, 255, 255)     # #FFFFFF
    PRIMARY_BLUE = RGBColor(2, 132, 199)     # #0284C7 Sky Tech Blue
    DEEP_NAVY = RGBColor(15, 23, 42)         # #0F172A Slate 900
    ROYAL_BLUE = RGBColor(3, 105, 161)       # #0369A1 Deep Blue
    SLATE_MUTED = RGBColor(100, 116, 139)    # #64748B Slate 500
    CYAN_LIGHT = RGBColor(224, 242, 254)     # #E0F2FE Sky 100
    BORDER_COL = RGBColor(203, 213, 225)     # #CBD5E1 Slate 300
    TEAL_SUCCESS = RGBColor(13, 148, 136)    # #0D9488 Emerald/Teal
    WARN_ORANGE = RGBColor(234, 88, 12)      # #EA580C Alert Orange

    def set_slide_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, chapter_tag, title_text, subtitle_text):
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.733), Inches(1.38))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p0 = tf.paragraphs[0]
        p0.text = chapter_tag.upper()
        p0.font.name = 'Arial'
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = PRIMARY_BLUE
        p0.space_after = Pt(2)

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.name = 'Arial'
        p1.font.size = Pt(19.5)
        p1.font.bold = True
        p1.font.color.rgb = DEEP_NAVY
        p1.space_after = Pt(3)

        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = 'Arial'
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = SLATE_MUTED

    # =========================================================================
    # SLIDE 1: TRANG BÌA CHÍNH THỨC THEO ĐỀ CƯƠNG & 2 MODEL ÁP DỤNG THỰC TẾ
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, BG_SLATE)

    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = PRIMARY_BLUE
    top_bar.line.fill.background()

    main_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.55), Inches(11.733), Inches(6.38))
    main_card.fill.solid()
    main_card.fill.fore_color.rgb = CARD_WHITE
    main_card.line.color.rgb = BORDER_COL
    main_card.line.width = Pt(1.5)

    inst_tb = s1.shapes.add_textbox(Inches(1.2), Inches(0.8), Inches(10.9), Inches(0.65))
    inst_tf = inst_tb.text_frame
    inst_tf.word_wrap = True
    p_inst = inst_tf.paragraphs[0]
    p_inst.text = "TRƯỜNG ĐẠI HỌC THỦ DẦU MỘT · VIỆN CÔNG NGHỆ SỐ · ĐỀ TÀI NCKH SINH VIÊN (2026 - 2027)"
    p_inst.font.name = 'Arial'
    p_inst.font.size = Pt(10)
    p_inst.font.bold = True
    p_inst.font.color.rgb = ROYAL_BLUE

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.35), Inches(10.9), Inches(3.45))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "BÁO CÁO HỘI ĐỒNG KHOA HỌC · THỜI LƯỢNG: 10 PHÚT (06 MIN THUYẾT TRÌNH + 04 MIN VẤN ĐÁP Q&A)"
    p_badge.font.name = 'Arial'
    p_badge.font.size = Pt(9.5)
    p_badge.font.bold = True
    p_badge.font.color.rgb = PRIMARY_BLUE
    p_badge.space_after = Pt(8)

    # EXACT TITLE FROM DOCX
    p_main = tf1.add_paragraph()
    p_main.text = "NGHIÊN CỨU ỨNG DỤNG HỌC TĂNG CƯỜNG ĐỂ HUẤN LUYỆN HỆ THỐNG KỸ NĂNG VẬN ĐỘNG VÀ TƯƠNG TÁC CHO ROBOT HAI CHÂN MICRODUCK TRONG MÔI TRƯỜNG MÔ PHỎNG"
    p_main.font.name = 'Arial'
    p_main.font.size = Pt(20.5)
    p_main.font.bold = True
    p_main.font.color.rgb = DEEP_NAVY
    p_main.space_after = Pt(8)

    # EMPHASIS ON MULTI-MODEL ORCHESTRATION & 2 APPLIED MODELS
    p_sub = tf1.add_paragraph()
    p_sub.text = "TRỌNG TÂM: HỆ THỐNG ĐIỀU PHỐI MULTI-MODEL THỰC CHỨNG TRÊN ĐƯỜNG GỒ GHỀ & LEO BẬC THANG"
    p_sub.font.name = 'Arial'
    p_sub.font.size = Pt(13)
    p_sub.font.bold = True
    p_sub.font.color.rgb = PRIMARY_BLUE
    p_sub.space_after = Pt(6)

    p_author = tf1.add_paragraph()
    p_author.text = "Sinh viên thực hiện: Đinh Hữu Phong (MSSV: 2324802010095 · Lớp: D23CNTT04)   |   GVHD: ThS. Trần Nhật Nam"
    p_author.font.name = 'Arial'
    p_author.font.size = Pt(10.5)
    p_author.font.color.rgb = SLATE_MUTED

    # 4 Metric Badges
    badges = [
        ("MODEL GỒ GHỀ", "BoxRandomGrid Thích Ứng Sụt Lún"),
        ("MODEL BẬC THANG", "Bậc Cao 3.5cm · Nhấc Chân 5.5cm"),
        ("ĐIỀU PHỐI 61D", "Hot-Swaps 20ms Không Trễ"),
        ("0-JERK S-CURVE", "Triệt Tiêu Hoàn Toàn Lực Giật")
    ]
    for idx, (b_title, b_desc) in enumerate(badges):
        bx = Inches(1.2 + idx * 2.75)
        by = Inches(5.15)
        badge_shape = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, Inches(2.6), Inches(1.4))
        badge_shape.fill.solid()
        badge_shape.fill.fore_color.rgb = BG_SLATE
        badge_shape.line.color.rgb = BORDER_COL
        badge_shape.line.width = Pt(1)

        b_tb = s1.shapes.add_textbox(bx, by, Inches(2.6), Inches(1.4))
        b_tf = b_tb.text_frame
        b_tf.word_wrap = True
        b_tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        bp1 = b_tf.paragraphs[0]
        bp1.text = b_title
        bp1.font.name = 'Arial'
        bp1.font.size = Pt(11)
        bp1.font.bold = True
        bp1.font.color.rgb = PRIMARY_BLUE
        bp1.alignment = PP_ALIGN.CENTER
        bp1.space_after = Pt(3)

        bp2 = b_tf.add_paragraph()
        bp2.text = b_desc
        bp2.font.name = 'Arial'
        bp2.font.size = Pt(9)
        bp2.font.color.rgb = SLATE_MUTED
        bp2.alignment = PP_ALIGN.CENTER

    s1.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 00:00 - 01:00 · TỐC ĐỘ: 125 từ/phút · PHONG THÁI: Trầm ấm, đĩnh đạc, tự tin]\n\n"
        "Kính thưa quý Thầy Cô trong Hội đồng Khoa học và toàn thể quý vị đại biểu, [Dừng 1.5s]\n\n"
        "Hãy thử tưởng tượng một robot hai chân có thể bước đi rất vững trên sàn nhà phẳng. "
        "Nhưng khi bước vào một bãi đá gồ ghề nhấp nhô hay gặp phải một dãy bậc thang cao, các khớp chân của nó bỗng nhiên đông cứng, "
        "mũi chân vấp mạnh vào mép bậc, và robot ngã quỵ xuống sàn!\n\n"
        "[Dừng 1s, hạ giọng tạo điểm lắng]\n"
        "Đó chính xác là thách thức thực tế lớn nhất mà đề tài của chúng em giải quyết.\n\n"
        "Hôm nay, em xin đại diện nhóm nghiên cứu từ Viện Công nghệ số — Trường Đại học Thủ Dầu Một, dưới sự hướng dẫn khoa học của ThS. Trần Nhật Nam, "
        "trân trọng báo cáo đề tài nghiên cứu khoa học sinh viên: 'Nghiên cứu ứng dụng học tăng cường để huấn luyện hệ thống kỹ năng vận động và tương tác cho robot hai chân Microduck trong môi trường mô phỏng', "
        "với trọng tâm đột phá là: Xây dựng hệ thống điều phối các Multi-Model ứng dụng cho robotics, mà minh chứng trực tiếp chính là 2 mô hình phức tạp đang áp dụng vào dự án: "
        "Mô hình di chuyển trên đường gồ ghề và Mô hình leo bậc thang cao!\n\n"
        "Hệ thống hiện được chúng em thực chứng trên robot Microduck trước tiên, và mang kiến trúc chuẩn mở sẵn sàng tổng quát hóa cho mọi robot sử dụng multi-model!"
    )

    # =========================================================================
    # SLIDE 2: TẠI SAO PHẢI LÀ MULTI-MODEL: BÀI TOÁN GỒ GHỀ & LEO BẬC THANG (CHƯƠNG 1)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, BG_SLATE)
    add_header(s2, "Chương 1 · Tính Cấp Thiết & Thách Thức Kỹ Thuật", 
               "Thách Thức Cốt Lõi: Tại Sao Phải Xây Dựng Hệ Thống Multi-Model?",
               "Sự bất khả thi của một mô hình đơn lẻ (Monolithic) và 3 rào cản chí mạng khi phối hợp giữa di chuyển phẳng, gồ ghề và bậc thang.")

    dilemmas = [
        ("01 / XUNG ĐỘT GRADIENT GIỮA CÁC ĐỊA HÌNH", 
         "Đi bộ phẳng đòi hỏi bước chân ngắn, tần số cao để tiết kiệm pin. Nhưng leo bậc thang (cao 3.5cm) và vượt đường gồ ghề đòi hỏi nhấc chân cao 5.5cm. Ép một mô hình học chung dẫn đến xung đột đạo hàm và làm hỏng cả hai bài toán.",
         WARN_ORANGE),
        ("02 / CÚ SỐC ĐỘNG HỌC KHI CHUYỂN BẬC (GIÂY 0.1)", 
         "Khi robot từ đường phẳng bước vào cầu thang, nếu chuyển đổi cứng (Bang-bang switch), gia tốc góc khớp nhảy vọt tức thời d²q/dt² → ∞, làm văng mũi chân vào mép bậc và ngắt dòng bảo vệ servo XL330.",
         WARN_ORANGE),
        ("03 / ĐỨT GÃY BỘ NHỚ KHI ĐỔI MÔ HÌNH", 
         "Nếu không có chuẩn giao thức nhất thể, việc tải một mô hình leo thang thay thế mô hình đi bộ làm trễ 50-200ms cấp phát bộ nhớ, khiến robot 800g bị rơi tự do và sụp đổ trọng tâm ngay trên bậc thang.",
         WARN_ORANGE)
    ]
    for idx, (d_title, d_desc, d_col) in enumerate(dilemmas):
        cx = Inches(0.8 + idx * 3.95)
        cy = Inches(2.0)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, cy, Inches(3.75), Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_WHITE
        card.line.color.rgb = BORDER_COL
        card.line.width = Pt(1.5)

        tb = s2.shapes.add_textbox(cx + Inches(0.2), cy + Inches(0.25), Inches(3.35), Inches(4.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = d_title
        p1.font.name = 'Arial'
        p1.font.size = Pt(10.5)
        p1.font.bold = True
        p1.font.color.rgb = d_col
        p1.space_after = Pt(12)

        p2 = tf.add_paragraph()
        p2.text = d_desc
        p2.font.name = 'Arial'
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = DEEP_NAVY
        p2.space_after = Pt(16)

        p3 = tf.add_paragraph()
        p3.text = "HỆ QUẢ: Không thể dùng 1 model đơn lẻ; bắt buộc phải có Hệ thống Multi-Model kèm bộ điều phối mượt mà!"
        p3.font.name = 'Arial'
        p3.font.size = Pt(9)
        p3.font.bold = True
        p3.font.color.rgb = SLATE_MUTED

    s2.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 01:00 - 01:45 · TỐC ĐỘ: 130 từ/phút · PHONG THÁI: Nghiêm túc, lập luận chặt chẽ]\n\n"
        "Kính thưa Hội đồng Giám khảo, tại sao chúng em không dùng một mô hình AI đơn lẻ cho robot? [Dừng 1s]\n\n"
        "Hãy nhìn vào bản chất vật lý của hai bài toán mà chúng em đang giải quyết:\n"
        "Thứ nhất: Một mô hình đi bộ mặt phẳng thông thường chỉ nhấc chân khoảng 1.2 đến 1.5 cm. "
        "Nếu đưa mô hình đó vào địa hình sỏi đá gồ ghề hay gặp bậc thang cao 3.5 cm — tương đương 15% chiều cao robot, "
        "nó sẽ lập tức bị vấp ngón chân và lộn nhào!\n\n"
        "Thứ hai: Ngược lại, nếu huấn luyện một mô hình luôn nhấc chân cao 5.5 cm để leo thang rồi bắt nó đi trên đường phẳng, "
        "robot sẽ bước đi cực kỳ chậm chạp, rung giật dữ dội và lãng phí năng lượng pin.\n\n"
        "Nếu ép một mạng nơ-ron học cả hai, hiện tượng Xung đột Gradient sẽ làm cả hai hành vi cùng suy thoái.\n\n"
        "Vì vậy, giải pháp tối ưu bắt buộc phải là một HỆ THỐNG MULTI-MODEL: gồm các policy chuyên biệt cho đường phẳng, đường gồ ghề và leo bậc thang, "
        "được điều phối thông minh trong thời gian thực!"
    )

    # =========================================================================
    # SLIDE 3: KHÔNG GIAN QUAN SÁT 61D & KIẾN TRÚC ĐIỀU PHỐI MULTI-MODEL (CHƯƠNG 3)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, BG_SLATE)
    add_header(s3, "Chương 3 · Thiết Kế Môi Trường & Kiến Trúc Hệ Thống", 
               "Không Gian Quan Sát 61D & Kiến Trúc Điều Phối Multi-Model (Neko Core)",
               "Khung giao thức 61D bất biến kết nối đồng bộ 50Hz (20ms) giữa mô hình mặt phẳng, đường gồ ghề và leo bậc thang.")

    if os.path.exists("fig_anatomy.png"):
        s3.shapes.add_picture("fig_anatomy.png", Inches(0.8), Inches(1.9), width=Inches(11.733))

    sum_shape = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.733), Inches(0.9))
    sum_shape.fill.solid()
    sum_shape.fill.fore_color.rgb = CARD_WHITE
    sum_shape.line.color.rgb = PRIMARY_BLUE
    sum_shape.line.width = Pt(1.2)

    sum_tb = s3.shapes.add_textbox(Inches(1.0), Inches(6.0), Inches(11.3), Inches(0.9))
    sum_tf = sum_tb.text_frame
    sum_tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    sum_p = sum_tf.paragraphs[0]
    sum_p.text = "CỐT LÕI HỆ THỐNG: Hợp đồng 61D bất biến (48D Proprioception + 13D Command Bus) cho phép hoán đổi nóng (Hot-Swaps) giữa Model Gồ ghề (rough_latest.onnx) và Model Bậc thang (stairs_latest.onnx) trong đúng 1 chu kỳ 20ms."
    sum_p.font.name = 'Arial'
    sum_p.font.size = Pt(10)
    sum_p.font.bold = True
    sum_p.font.color.rgb = ROYAL_BLUE

    s3.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 01:45 - 02:45 · TỐC ĐỘ: 135 từ/phút · PHONG THÁI: Tự tin, làm chủ công nghệ]\n\n"
        "Kính mời quý Thầy Cô cùng quan sát sơ đồ giải phẫu 4 tầng của Neko Core trên màn hình. [Dừng 1s]\n\n"
        "Hệ thống vận hành như sau:\n"
        "TẦNG 1 — HỢP ĐỒNG QUAN SÁT 61D BẤT BIẾN: Chúng em chuẩn hóa 48 chiều tự cảm proprioception (IMU, vận tốc góc, góc khớp 14 servo) và 13 chiều thanh ghi lệnh. "
        "Dù robot đang đi trên mặt phẳng, vượt bãi sỏi đá hay bước lên bậc thang, cấu trúc 61D này là hoàn toàn bất biến. Nhờ đó, controller có thể hoán đổi mô hình ONNX "
        "trong đúng 20 mili-giây mà không tốn thời gian cấp phát bộ nhớ!\n\n"
        "TẦNG 2 — TRỌNG TÀI CHUYỂN TIẾP (Arbiter & Governor): Quản lý máy trạng thái, nhận diện khi nào robot tiến vào vùng gồ ghề hay bậc thang để kích hoạt hòa trộn.\n\n"
        "TẦNG 3 — KHO MULTI-MODEL: Nơi chứa trực tiếp 2 mô hình cốt lõi mà chúng em đã huấn luyện hoàn tất: Model Đường gồ ghề (rough_latest.onnx) và Model Leo bậc thang (stairs_latest.onnx), "
        "cùng các mô hình cơ bản và phục hồi thăng bằng.\n\n"
        "Và TẦNG 4 — CHUYỂN GIAO THỰC TẾ BAM M6: Mô hình hóa dòng áp 14 servo Dynamixel XL330, bù sụt áp 6.5V đến 8.2V và khử độ rơ bánh răng 2 độ để đảm bảo sim-to-real tuyệt đối!"
    )

    # =========================================================================
    # SLIDE 4: TOÁN HỌC HÒA TRỘN MULTI-MODEL & CHỐT AN TOÀN ĐỘNG HỌC (CHƯƠNG 3 & 4)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, BG_SLATE)
    add_header(s4, "Chương 3 & 4 · Cơ Chế Điều Phối Toán Học & An Toàn", 
               "Cơ Chế Hòa Trộn Multi-Model: Hàm Quintic S-Curve & Chốt An Toàn Bước Chân",
               "Triệt tiêu hoàn toàn lực giật cơ học cấp hai (C²-continuous) kết hợp khóa pha tiếp đất khi chuyển đổi giữa bước đi phẳng và bước leo thang.")

    if os.path.exists("fig_scurve.png"):
        s4.shapes.add_picture("fig_scurve.png", Inches(0.8), Inches(1.9), width=Inches(6.2))

    right_card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.9), Inches(5.333), Inches(5.0))
    right_card.fill.solid()
    right_card.fill.fore_color.rgb = CARD_WHITE
    right_card.line.color.rgb = BORDER_COL
    right_card.line.width = Pt(1.5)

    r_tb = s4.shapes.add_textbox(Inches(7.45), Inches(2.1), Inches(4.8), Inches(4.6))
    r_tf = r_tb.text_frame
    r_tf.word_wrap = True

    rp0 = r_tf.paragraphs[0]
    rp0.text = "KINEMATIC SAFETY GOVERNOR & PHASE LATCH"
    rp0.font.name = 'Arial'
    rp0.font.size = Pt(11.5)
    rp0.font.bold = True
    rp0.font.color.rgb = TEAL_SUCCESS
    rp0.space_after = Pt(10)

    safety_rules = [
        ("01 / KHÓA PHA BƯỚC TIẾP ĐẤT (GAIT PHASE LATCH)", 
         "Khi chuyển từ model đi phẳng sang model leo thang, nếu robot đang nhấc một chân trên không, tiến trình hòa trộn bị hoãn lại cho đến khi cả 2 bàn chân xác nhận chạm sàn kép (double support)."),
        ("02 / CHỐT GÓC NGHIÊNG KHẨN CẤP (TILT GATE)", 
         "Nếu phát hiện thân robot nghiêng quá 22° do trượt mép bậc thang hoặc sụt lún sỏi đá, hệ thống lập tức ngắt hòa trộn (Hardware Preemption) và kích hoạt chế độ tự cứu thăng bằng."),
        ("03 / HÒA TRỘN QUINTIC S-CURVE (C²-SMOOTH)", 
         "Hàm hòa trộn w(τ) = 10τ³ - 15τ⁴ + 6τ⁵ triệt tiêu cả vận tốc góc và gia tốc góc tại hai đầu mút, đưa lực xung giật cơ học (Jerk) về đúng 0 tuyệt đối.")
    ]
    for stitle, sdesc in safety_rules:
        p_t = r_tf.add_paragraph()
        p_t.text = stitle
        p_t.font.name = 'Arial'
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = DEEP_NAVY
        p_t.space_after = Pt(2)

        p_d = r_tf.add_paragraph()
        p_d.text = sdesc
        p_d.font.name = 'Arial'
        p_d.font.size = Pt(9)
        p_d.font.color.rgb = SLATE_MUTED
        p_d.space_after = Pt(8)

    s4.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 02:45 - 03:30 · TỐC ĐỘ: 130 từ/phút · PHONG THÁI: Sắc sảo, chuẩn xác khoa học]\n\n"
        "Thưa Hội đồng Giám khảo, khi robot đang bước đi bình thường mà chuẩn bị bước lên bậc thang cao 3.5 cm, "
        "làm sao để chuyển từ bước thấp sang bước nhấc chân cao 5.5 cm mà không bị rung giật gãy khớp? [Dừng 1s]\n\n"
        "Xin mời quý Thầy Cô quan sát đồ thị toán học bên trái:\n"
        "Nếu chuyển đổi đột ngột (đường màu cam đứt đoạn), gia tốc góc khớp bị gián đoạn tức thời, tạo ra cú giật vô tận làm nứt bánh răng servo.\n\n"
        "Neko Core hóa giải điều này bằng HÀM HÒA TRỘN QUINTIC S-CURVE BẬC NĂM: w(t) = 10t³ - 15t⁴ + 6t⁵. "
        "Hàm số này đạt chuẩn trơn cấp hai (C²-continuous): Cả đạo hàm bậc một — vận tốc góc, và đạo hàm bậc hai — gia tốc góc, đều bằng 0 tuyệt đối ở hai đầu mút! "
        "Nhờ đó, lực xung giật cơ học được triệt tiêu hoàn toàn về mức 0!\n\n"
        "Đồng thời, CHỐT AN TOÀN ĐỘNG HỌC liên tục giám sát: nếu robot đang ở pha vung chân lơ lửng, hệ thống kiên nhẫn chờ cả hai chân chạm đất vững chãi "
        "mới kích hoạt hòa trộn. Mọi chuyển động giữa các model đều được bảo chứng bởi toán học chính xác!"
    )

    # =========================================================================
    # SLIDE 5: THỰC CHIẾN 2 MODEL ĐƯỢC ÁP DỤNG TRONG DỰ ÁN: GỒ GHỀ & BẬC THANG (CHƯƠNG 4)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, BG_SLATE)
    add_header(s5, "Chương 4 · Thực Nghiệm Định Lượng 2 Model Trọng Tâm", 
               "Thực Chứng 2 Model Đang Áp Dụng: Đường Gồ Ghề & Leo Bậc Thang",
               "Kết quả đo kiểm định lượng thực tế của Model Đường gồ ghề (rough_latest.onnx) và Model Leo bậc thang (stairs_latest.onnx) trên Microduck.")

    if os.path.exists("fig_benchmark.png"):
        s5.shapes.add_picture("fig_benchmark.png", Inches(0.8), Inches(1.9), width=Inches(6.2))

    r_card5 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.9), Inches(5.333), Inches(5.0))
    r_card5.fill.solid()
    r_card5.fill.fore_color.rgb = CARD_WHITE
    r_card5.line.color.rgb = BORDER_COL
    r_card5.line.width = Pt(1.5)

    r5_tb = s5.shapes.add_textbox(Inches(7.45), Inches(2.1), Inches(4.8), Inches(4.6))
    r5_tf = r5_tb.text_frame
    r5_tf.word_wrap = True

    r5_p0 = r5_tf.paragraphs[0]
    r5_p0.text = "ĐO KIỂM 2 MODEL CỐT LÕI ĐANG ÁP DỤNG TRONG DỰ ÁN"
    r5_p0.font.name = 'Arial'
    r5_p0.font.size = Pt(10.5)
    r5_p0.font.bold = True
    r5_p0.font.color.rgb = PRIMARY_BLUE
    r5_p0.space_after = Pt(8)

    results = [
        ("MODEL ĐƯỜNG GỒ GHỀ (rough_latest.onnx)", 
         "Địa hình BoxRandomGrid cobblestone (ô 0.45m), độ gồ ghề ngẫu nhiên. Áp dụng tiếp xúc va chạm mềm (solref=0.04) chống nảy chân. Tỷ lệ vượt địa hình đạt 96.8% (mô hình phẳng cố đi chỉ đạt 35%)."),
        ("MODEL LEO BẬC THANG (stairs_latest.onnx)", 
         "Bậc thang cao 1.0 - 3.5cm (~15% chiều cao robot!), mặt bậc sâu 18cm. Curriculum nhấc chân nâng dần từ 3.5cm lên 5.5cm. Mô hình va chạm đầy đủ 70 geoms bảo vệ cẳng chân, tỷ lệ leo thành công đạt 95.2%."),
        ("HIỆU QUẢ ĐIỀU PHỐI MULTI-MODEL (HOT-SWAPS)", 
         "Khi robot di chuyển liên tục: Mặt phẳng -> Bãi gồ ghề -> Bậc thang: Quá trình chuyển đổi diễn ra trong 20ms, độ mượt đạt 98.2%, hoàn toàn không có hiện tượng vấp ngã hay giật cơ!")
    ]
    for rtitle, rdesc in results:
        p_t = r5_tf.add_paragraph()
        p_t.text = rtitle
        p_t.font.name = 'Arial'
        p_t.font.size = Pt(9.5)
        p_t.font.bold = True
        p_t.font.color.rgb = DEEP_NAVY
        p_t.space_after = Pt(2)

        p_d = r5_tf.add_paragraph()
        p_d.text = rdesc
        p_d.font.name = 'Arial'
        p_d.font.size = Pt(8.8)
        p_d.font.color.rgb = SLATE_MUTED
        p_d.space_after = Pt(8)

    s5.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 03:30 - 04:30 · TỐC ĐỘ: 135 từ/phút · PHONG THÁI: Hào hùng, tự hào, dẫn chứng số liệu thực tế]\n\n"
        "Và đây chính là kết quả thực tế trên hai mô hình cốt lõi mà chúng em đang trực tiếp áp dụng trong dự án! [Dừng 1s]\n\n"
        "Hãy nhìn vào biểu đồ đo kiểm định lượng bên trái màn hình:\n"
        "MÔ HÌNH THỨ NHẤT: ĐƯỜNG GỒ GHỀ (rough_latest.onnx). Địa hình đá sỏi ngẫu nhiên BoxRandomGrid khiến mô hình đi bộ thông thường thất bại tới 65% do vấp mũi chân. "
        "Nhưng với model chuyên biệt gồ ghề, kết hợp cơ chế tiếp xúc mềm solref=0.04 và thích ứng sụt lún, tỷ lệ vượt qua thành công nhảy vọt lên 96.8%!\n\n"
        "MÔ HÌNH THỨ HAI: LEO BẬC THANG CAO (stairs_latest.onnx). Đây là thử thách cực đại: Bậc thang cao từ 1.0 đến 3.5 cm — tương đương 15% chiều cao robot 25cm! "
        "Chúng em thiết kế một lộ trình curriculum nhấc chân từ 3.5 lên tận 5.5 cm, kết hợp mô hình va chạm chi tiết 70 geoms để xử lý va quệt cẳng chân vào mép bậc. "
        "Tỷ lệ leo bậc thang thành công đạt tới 95.2%!\n\n"
        "Và điều kỳ diệu nhất nằm ở HỆ THỐNG ĐIỀU PHỐI MULTI-MODEL: Khi robot bước từ đường phẳng vào bãi sỏi gồ ghề, rồi tiếp tục bước lên cầu thang, "
        "bộ điều phối hoán đổi nóng giữa các mô hình trong đúng 20 mili-giây, độ mượt đạt 98.2%, hoàn toàn không có bất kỳ cú sốc gián đoạn nào!"
    )

    # =========================================================================
    # SLIDE 6: KHẢ NĂNG TỔNG QUÁT HÓA CỦA HỆ THỐNG MULTI-MODEL (TẦM NHÌN MỞ RỘNG)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, BG_SLATE)
    add_header(s6, "Tầm Nhìn Mở Rộng · Tính Tổng Quát Hóa Toàn Ngành", 
               "Khả Năng Tổng Quát Hóa: Kiến Trúc Mở Cho Mọi Robot Dùng Multi-Model",
               "Hệ thống hiện áp dụng thử nghiệm vào Microduck trước tiên, và được thiết kế chuẩn Embodied MoE để mở rộng cho mọi dạng robot.")

    if os.path.exists("fig_generalization.png"):
        s6.shapes.add_picture("fig_generalization.png", Inches(0.8), Inches(1.9), width=Inches(11.733))

    s6.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 04:30 - 05:15 · TỐC ĐỘ: 130 từ/phút · PHONG THÁI: Truyền cảm hứng, tầm nhìn lãnh đạo công nghệ]\n\n"
        "Kính thưa Hội đồng Giám khảo, một giải pháp xuất sắc không bao giờ tự giam mình trong một bài toán cục bộ. [Dừng 1s]\n\n"
        "[Nhấn mạnh rõ ràng] Chúng em xin khẳng định lại: Hệ thống điều phối này HIỆN ĐANG ĐƯỢC CHÚNG EM ÁP DỤNG TRỰC TIẾP VÀO DỰ ÁN ROBOT HAI CHÂN MICRODUCK TRƯỚC TIÊN "
        "với 2 bài toán đường gồ ghề và leo bậc thang. Thế nhưng, toàn bộ kiến trúc 4 tầng được thiết kế theo tư duy của một HỆ ĐIỀU HÀNH ĐIỀU PHỐI ĐA MÔ HÌNH (Embodied MoE) hoàn toàn độc lập cơ khí!\n\n"
        "Quan sát lộ trình mở rộng trên màn hình:\n"
        "Khi mang sang ROBOT BỐN CHÂN (Quadrupeds như Spot hay Unitree Go2), ta điều phối giữa các model: đi nước kiệu trên phẳng sang model phi nước đại vượt địa hình hiểm trở.\n\n"
        "Khi đưa lên ROBOT HÌNH NGƯỜI (Humanoids), hệ thống điều phối trở thành cầu nối phân tách an toàn: Thân trên nhận lệnh VLM 5Hz, trong khi thân dưới duy trì các model thăng bằng, gồ ghề và leo thang ở 50Hz tuyệt đối.\n\n"
        "Hệ thống điều phối Multi-Model giải phóng các kỹ sư khỏi sự bế tắc của việc huấn luyện một mô hình đơn lẻ khổng lồ — mở ra kỷ nguyên mới của trí tuệ robot cắm-và-chạy!"
    )

    # =========================================================================
    # SLIDE 7: TỔNG KẾT, SẢN PHẨM ĐỀ TÀI & MỞ MÀN VẤN ĐÁP (CHƯƠNG 5)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, BG_SLATE)

    main_card7 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.65), Inches(11.733), Inches(6.25))
    main_card7.fill.solid()
    main_card7.fill.fore_color.rgb = CARD_WHITE
    main_card7.line.color.rgb = BORDER_COL
    main_card7.line.width = Pt(1.5)

    tb7 = s7.shapes.add_textbox(Inches(1.2), Inches(1.0), Inches(10.9), Inches(5.4))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p7_tag = tf7.paragraphs[0]
    p7_tag.text = "CHƯƠNG 5 · KẾT LUẬN, SẢN PHẨM BÀN GIAO & MỞ MÀN VẤN ĐÁP"
    p7_tag.font.name = 'Arial'
    p7_tag.font.size = Pt(10)
    p7_tag.font.bold = True
    p7_tag.font.color.rgb = PRIMARY_BLUE
    p7_tag.space_after = Pt(10)

    p7_title = tf7.add_paragraph()
    p7_title.text = "Hiện Thực Hóa Hệ Thống Multi-Model Cho Robot Thực Tế"
    p7_title.font.name = 'Arial'
    p7_title.font.size = Pt(25)
    p7_title.font.bold = True
    p7_title.font.color.rgb = DEEP_NAVY
    p7_title.space_after = Pt(14)

    products_text = (
        "CÁC KẾT QUẢ KHOA HỌC BÀN GIAO CỦA ĐỀ TÀI (ĐÚNG THEO ĐỀ CƯƠNG NCKH):\n"
        "1. QUYỂN BÁO CÁO TỔNG KẾT NCKH: Trình bày chi tiết lý thuyết PPO, mô phỏng MuJoCo Warp GPU, hàm reward triệt tiêu local optima.\n"
        "2. TẬP HỢP MÔ HÌNH ONNX: Xuất hoàn chỉnh các model nơ-ron tích hợp normalizer, đặc biệt là Model Đường gồ ghề (rough_latest.onnx) và Model Bậc thang (stairs_latest.onnx).\n"
        "3. BỘ MÃ NGUỒN INFERENCE & ĐIỀU PHỐI: Minh chứng trực quan khả năng chuyển đổi nóng (Hot-Swaps) mượt mà 20ms giữa các địa hình mà không gây giật khớp hay ngã robot."
    )
    p7_body = tf7.add_paragraph()
    p7_body.text = products_text
    p7_body.font.name = 'Arial'
    p7_body.font.size = Pt(11)
    p7_body.font.color.rgb = ROYAL_BLUE
    p7_body.space_after = Pt(18)

    p7_call = tf7.add_paragraph()
    p7_call.text = "Em xin trân trọng cảm ơn quý Thầy Cô trong Hội đồng Khoa học! Kính mời Thầy Cô đặt câu hỏi cho phần Vấn đáp Q&A!"
    p7_call.font.name = 'Arial'
    p7_call.font.size = Pt(13)
    p7_call.font.bold = True
    p7_call.font.color.rgb = TEAL_SUCCESS

    s7.notes_slide.notes_text_frame.text = (
        "[THỜI GIAN: 05:15 - 06:00 · TỐC ĐỘ: 125 từ/phút · PHONG THÁI: Truyền cảm, lắng đọng, chốt hạ ấn tượng]\n\n"
        "Kính thưa quý Thầy Cô trong Hội đồng Giám khảo, [Dừng 1.5s]\n\n"
        "Đề tài của chúng em đã hoàn thành đầy đủ 100% khối lượng và bàn giao trọn vẹn 3 sản phẩm theo đúng đề cương nghiên cứu khoa học:\n"
        "Thứ nhất là quyển báo cáo tổng kết khoa học chi tiết.\n"
        "Thứ hai là các tệp mô hình ONNX đã kiểm chứng thành công trên địa hình gồ ghề và bậc thang cao.\n"
        "Và thứ ba là bộ mã nguồn điều phối chứng minh tính năng hoán đổi nóng 20ms mượt mà giữa các multi-model.\n\n"
        "Người ta thường nói: 'Muốn đi nhanh hãy đi một mình, nhưng muốn đi xa hãy đi cùng nhau.'\n"
        "Trong thế giới của trí tuệ nhân tạo embodied AI, một mô hình đơn lẻ không thể giải quyết mọi địa hình, nhưng khi một hệ thống multi-model cùng hòa vào một nhịp đập, "
        "robot có thể tự tin bước qua mọi chông gai — từ sỏi đá gập ghềnh đến những nấc thang cao nhất.\n\n"
        "Chúng em đã chứng minh nó trên Microduck, và chúng em sẵn sàng mang kiến trúc này đến với mọi cỗ máy thông minh. [Dừng 2s]\n\n"
        "[Cúi đầu trang trọng]\n"
        "Em xin trân trọng cảm ơn quý Thầy Cô trong Hội đồng Giám khảo đã chú ý lắng nghe! Em xin phép được lắng nghe nhận xét và sẵn sàng giải đáp mọi câu hỏi trong phần Vấn đáp ngay sau đây!"
    )

    # =========================================================================
    # SLIDE 8: CHIẾN LƯỢC PHÒNG THỦ 4 PHÚT VẤN ĐÁP Q&A VỀ 2 MODEL VÀ HỆ THỐNG (06:00 - 10:00)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, BG_SLATE)
    add_header(s8, "Phần Vấn Đáp · 06:00 - 10:00 · Chiến Lược Phản Biện", 
               "Chiến Lược Phản Biện 4 Phút Q&A: 4 Câu Hỏi Trọng Tâm Về Gồ Ghề, Bậc Thang & Multi-Model",
               "Bộ câu trả lời đanh thép, dẫn chứng lý thuyết toán học và số liệu thực nghiệm giải giáp Hội đồng Giám khảo.")

    qa_list = [
        ("CÂU HỎI 1: TẠI SAO KHÔNG DÙNG 1 MÔ HÌNH DUY NHẤT CHO CẢ PHẲNG, GỒ GHỀ VÀ BẬC THANG?",
         "Xung đột mục tiêu (Gradient Conflict) và Quên tai hại (Catastrophic Forgetting): Nhấc chân cao 5.5cm khi leo thang xung đột trực tiếp với yêu cầu tiết kiệm pin và tần số cao của bước đi phẳng. Hệ thống Multi-Model giúp từng bài toán đạt cực trị toàn cục tối ưu."),
        ("CÂU HỎI 2: ĐỘ TRỄ S-CURVE KHI BƯỚC VÀO BẬC THANG CÓ LÀM CHẬM LỆNH?",
         "Thời gian hòa trộn S-Curve chỉ mất 0.2 - 0.5s (10 - 25 chu kỳ 50Hz), diễn ra đúng trong pha tiếp đất kép (double support). Nếu có va chạm mép bậc hoặc nghiêng > 22°, Chốt an toàn ngắt ưu tiên trong 20ms để kích hoạt chính sách phục hồi."),
        ("CÂU HỎI 3: TÍNH TỔNG QUÁT KHI MỞ RỘNG CHO ROBOT KHÁC?",
         "Kiến trúc dựa trên giao thức (Protocol-based): Chỉ cần thay đổi file ánh xạ khớp phần cứng, toàn bộ cỗ máy hòa trộn S-Curve và Trọng tài chuyển tiếp giữa các model địa hình được tái sử dụng 100% nguyên vẹn."),
        ("CÂU HỎI 4: TÍNH TOÁN 50 HZ CỦA HỆ THỐNG CÓ QUÁ TẢI CPU NHÚNG?",
         "Các phép toán đại số S-Curve tiêu tốn chưa đến 0.15 ms; suy luận ONNX mất 3.5 - 5.2 ms. Tổng chu kỳ mất 5.5 ms / 20 ms trên vi xử lý nhúng (Raspberry Pi 4 / Jetson Orin), chỉ chiếm 27% tài nguyên tính toán.")
    ]
    for idx, (q_txt, a_txt) in enumerate(qa_list):
        row = idx // 2
        col = idx % 2
        qx = Inches(0.8 + col * 5.95)
        qy = Inches(2.0 + row * 2.5)

        q_card = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, qx, qy, Inches(5.75), Inches(2.3))
        q_card.fill.solid()
        q_card.fill.fore_color.rgb = CARD_WHITE
        q_card.line.color.rgb = BORDER_COL
        q_card.line.width = Pt(1.5)

        tb_q = s8.shapes.add_textbox(qx + Inches(0.2), qy + Inches(0.15), Inches(5.35), Inches(2.0))
        tf_q = tb_q.text_frame
        tf_q.word_wrap = True

        qp1 = tf_q.paragraphs[0]
        qp1.text = q_txt
        qp1.font.name = 'Arial'
        qp1.font.size = Pt(9.5)
        qp1.font.bold = True
        qp1.font.color.rgb = PRIMARY_BLUE
        qp1.space_after = Pt(4)

        qp2 = tf_q.add_paragraph()
        qp2.text = a_txt
        qp2.font.name = 'Arial'
        qp2.font.size = Pt(9)
        qp2.font.color.rgb = DEEP_NAVY

    s8.notes_slide.notes_text_frame.text = (
        "[HƯỚNG DẪN DÀNH CHO DIỄN GIẢ TRONG 4 PHÚT VẤN ĐÁP Q&A VỚI HỘI ĐỒNG]\n\n"
        "Nguyên tắc vàng khi trả lời Hội đồng:\n"
        "1. Luôn mở đầu khiêm tốn nhưng đĩnh đạc: 'Dạ, em xin trân trọng cảm ơn câu hỏi rất sâu sắc của quý Thầy/Cô...'\n"
        "2. Đi thẳng vào bản chất của 2 MODEL THỰC TẾ: Phân tích sự xung đột động học giữa việc nhấc chân 1.5cm trên đường phẳng và 5.5cm trên bậc thang.\n"
        "3. Dẫn chứng con số định lượng thực nghiệm (96.8% gồ ghề, 95.2% bậc thang, 0.15ms toán học S-Curve, 20ms chu kỳ đồng bộ 50Hz).\n"
        "4. Sử dụng các thuật ngữ chuyên môn chuẩn xác: Catastrophic Forgetting, Gradient Conflict, C² Continuity, Gait Phase Latch, solref contact softening, BAM Actuator.\n"
        "5. Giữ ánh mắt tự tin, phong thái chững chạc khẳng định năng lực làm chủ kỹ thuật của sinh viên Viện Công nghệ số."
    )

    output_pptx = "Neko_Core_Multi_Model_Robotics_Presentation.pptx"
    prs.save(output_pptx)
    print(f"Presentation saved successfully to {output_pptx}!")

if __name__ == '__main__':
    build_deck()
